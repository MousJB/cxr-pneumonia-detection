"""Training pieces of PROTOCOL.md: sampler (§3.4), warm-up and early stopping (§6), τ (§7.2)."""

import numpy as np
import pytest
import torch

from cxr.training import (
    TAU_GRID,
    BalancedEpochSampler,
    EarlyStopping,
    EpochIndexedDataset,
    select_tau,
    warmup_factor,
    xyxy_to_xywh,
)

TARGETS = np.array([1] * 30 + [0] * 100)  # 30 positives, 100 negatives


def test_sampler_epoch_content():
    s = BalancedEpochSampler(TARGETS, seed=1)
    assert len(s) == 60
    for epoch in range(5):
        idx = s.indices(epoch)
        assert len(idx) == 60
        assert sorted(idx[TARGETS[idx] == 1]) == list(range(30))  # every positive once
        assert (TARGETS[idx] == 0).sum() == 30


def test_sampler_negatives_cycle_without_repeat():
    s = BalancedEpochSampler(TARGETS, seed=1)
    first_cycle = np.concatenate([s.negatives(e) for e in range(3)])  # 90 of the 100 negatives
    assert len(set(first_cycle)) == 90
    two_cycles = np.concatenate([s.negatives(e) for e in range(20)])  # 600 draws = 6 permutations
    assert all(sorted(two_cycles[k * 100:(k + 1) * 100]) == list(range(30, 130)) for k in range(6))


def test_sampler_deterministic_and_epoch_dependent():
    a, b = BalancedEpochSampler(TARGETS, seed=1), BalancedEpochSampler(TARGETS, seed=1)
    assert np.array_equal(a.indices(2), b.indices(2))
    assert not np.array_equal(a.indices(2), a.indices(3))
    a.set_epoch(4)
    assert list(a) == [(4, int(i)) for i in a.indices(4)]


class EpochEcho:
    """Stands in for DetectionDataset: returns the epoch it sees and the index."""

    epoch = 0

    def __len__(self):
        return len(TARGETS)

    def __getitem__(self, i):
        return torch.tensor([self.epoch, i])


@pytest.mark.parametrize("workers", [0, 2])
def test_epoch_reaches_persistent_workers(workers):
    """With persistent workers, setting an attribute in the main process would not reach the
    workers; the epoch carried by the sampler index does."""
    sampler = BalancedEpochSampler(TARGETS, seed=1)
    loader = torch.utils.data.DataLoader(EpochIndexedDataset(EpochEcho()), batch_size=10, sampler=sampler,
                                         num_workers=workers, persistent_workers=workers > 0)
    for epoch in range(3):
        sampler.set_epoch(epoch)
        seen = torch.cat(list(loader))
        assert set(seen[:, 0].tolist()) == {epoch}
        assert sorted(seen[:, 1].tolist()) == sorted(sampler.indices(epoch).tolist())


def test_warmup_factor():
    assert warmup_factor(0) == pytest.approx(0.001)
    assert warmup_factor(250) == pytest.approx(0.001 + 0.999 * 0.5)
    assert warmup_factor(500) == 1.0
    assert warmup_factor(10_000) == 1.0


def test_early_stopping_patience_3():
    stopper = EarlyStopping(3)
    history = [stopper.step(v, e) for e, v in enumerate([0.10, 0.12, 0.12, 0.11, 0.13, 0.13, 0.12, 0.12], 1)]
    assert [h[0] for h in history] == [True, True, False, False, True, False, False, False]
    assert [h[1] for h in history] == [False] * 7 + [True]
    assert stopper.best_epoch == 5 and stopper.best == 0.13


def test_tau_grid():
    assert TAU_GRID[0] == 0.01 and TAU_GRID[1] == 0.05 and TAU_GRID[-1] == 0.95
    assert len(TAU_GRID) == 20


def test_select_tau_prefers_highest_tau_on_tie():
    truth = {"a": [[0, 0, 10, 10]], "b": []}
    preds = {"a": (np.array([[0, 0, 10, 10]], float), np.array([0.9])),
             "b": (np.array([[50, 50, 10, 10]], float), np.array([0.3]))}
    tau, score, per_tau = select_tau(preds, truth)
    # τ ≤ 0.30 keeps the false positive on b (score 0.5); 0.35 to 0.90 give 1.0; 0.95 drops the hit.
    assert per_tau[0.3] == 0.5 and per_tau[0.35] == 1.0 and per_tau[0.9] == 1.0
    assert tau == 0.9 and score == 1.0


def test_xyxy_to_xywh():
    np.testing.assert_array_equal(xyxy_to_xywh(np.array([[1, 2, 11, 22]])), [[1, 2, 10, 20]])
    assert xyxy_to_xywh(np.zeros((0, 4))).shape == (0, 4)
