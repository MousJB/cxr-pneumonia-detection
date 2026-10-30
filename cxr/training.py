"""Pieces of the final training defined by PROTOCOL.md: balanced sampler (§3.4), learning-rate
warm-up and early stopping (§6), threshold choice on validation (§7.2), inference."""

import numpy as np
import torch

from cxr.metric import dataset_score

# §7.2: 0.01, then 0.05 to 0.95 by 0.05 (20 values)
TAU_GRID = (0.01,) + tuple(round(0.05 * k, 2) for k in range(1, 20))


class BalancedEpochSampler(torch.utils.data.Sampler):
    """Yields ``(epoch, index)``: every positive image + as many negatives, shuffled.

    The negatives are taken in turn from seeded permutations of all negatives: within a
    permutation no image is repeated; when one is used up, the next one starts. The epoch is
    part of each index, so the data workers (even persistent ones) draw the augmentation of
    the right epoch. Call ``set_epoch`` before each epoch.
    """

    def __init__(self, targets, seed: int):
        targets = np.asarray(targets)
        self.pos = np.flatnonzero(targets == 1)
        self.neg = np.flatnonzero(targets == 0)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def negatives(self, epoch: int) -> np.ndarray:
        n = len(self.pos)
        start, stop = epoch * n, (epoch + 1) * n
        cycles = []
        for c in range(start // len(self.neg), (stop - 1) // len(self.neg) + 1):
            cycles.append(np.random.default_rng([self.seed, 1, c]).permutation(self.neg))
        offset = (start // len(self.neg)) * len(self.neg)
        return np.concatenate(cycles)[start - offset: stop - offset]

    def indices(self, epoch: int) -> np.ndarray:
        idx = np.concatenate([self.pos, self.negatives(epoch)])
        return np.random.default_rng([self.seed, 2, epoch]).permutation(idx)

    def __iter__(self):
        return iter([(self.epoch, int(i)) for i in self.indices(self.epoch)])

    def __len__(self):
        return 2 * len(self.pos)


class EpochIndexedDataset(torch.utils.data.Dataset):
    """Wraps a ``DetectionDataset``: ``dataset[(epoch, i)]`` draws the augmentation of that epoch."""

    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        epoch, i = item
        self.base.epoch = epoch  # per process copy: safe in the data workers
        return self.base[i]


def warmup_factor(step: int, warmup_steps: int = 500, start: float = 0.001) -> float:
    """Linear warm-up of the learning rate from ``start`` × lr to lr over ``warmup_steps`` steps."""
    if step >= warmup_steps:
        return 1.0
    return start + (1.0 - start) * step / warmup_steps


class EarlyStopping:
    """Stop after ``patience`` consecutive epochs without a strict increase of the metric."""

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.best = -np.inf
        self.best_epoch = None
        self.bad_epochs = 0

    def step(self, value: float, epoch: int) -> tuple[bool, bool]:
        """Returns ``(improved, stop)``."""
        if value > self.best:
            self.best, self.best_epoch, self.bad_epochs = value, epoch, 0
            return True, False
        self.bad_epochs += 1
        return False, self.bad_epochs >= self.patience

    def state_dict(self):
        return {"best": self.best, "best_epoch": self.best_epoch, "bad_epochs": self.bad_epochs}

    def load_state_dict(self, state):
        self.best, self.best_epoch, self.bad_epochs = state["best"], state["best_epoch"], state["bad_epochs"]


def above(predictions, tau: float):
    """Keeps the boxes with score ≥ tau. ``predictions[case] = (boxes_xywh (N, 4), scores (N,))``."""
    out = {}
    for case, (boxes, scores) in predictions.items():
        keep = scores >= tau
        out[case] = (boxes[keep], scores[keep])
    return out


def select_tau(predictions, ground_truth, grid=TAU_GRID) -> tuple[float, float, dict]:
    """τ of the grid that maximises the competition metric (highest τ on a tie).

    Returns ``(tau, score, {tau: score})``.
    """
    scores = {tau: dataset_score(above(predictions, tau), ground_truth)[0] for tau in grid}
    best_tau = max(grid, key=lambda t: (np.nan_to_num(scores[t], nan=-1.0), t))
    return best_tau, scores[best_tau], scores


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)


@torch.no_grad()
def predict(model, loader, device, amp: bool = False) -> dict:
    """``{case: (boxes_xywh, scores)}`` in 512 px coordinates (every box the model returns)."""
    model.eval()
    out = {}
    for images, _, cases in loader:
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            results = model([im.to(device, non_blocking=True) for im in images])
        for case, r in zip(cases, results):
            out[case] = (xyxy_to_xywh(r["boxes"].float().cpu().numpy()), r["scores"].float().cpu().numpy())
    return out
