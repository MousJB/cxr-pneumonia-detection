"""Reading the prepared data (``scripts/prepare_rsna.py``) for training and inference.

Augmentation transforms are plain callables
``transform(image_uint8: (H, W), boxes_xyxy: (N, 4), rng: Generator) -> (image, boxes)``.

Box convention (the one of torchvision): ``[x1, y1, x2, y2]`` in continuous pixel units,
pixel ``(row r, column c)`` covering ``[c, c + 1) × [r, r + 1)``, so a box that covers the
whole image is ``[0, 0, W, H]``. A transform may remove boxes (see ``Affine``) but never
reorders them; the dataset builds the labels from the boxes it gets back, so boxes and
labels stay aligned.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

MIN_BOX_SIDE = 1.0  # px; a box narrower or shorter than this after clipping is removed


def load_images() -> list[dict[str, str]]:
    with (DATA / "images.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_boxes() -> dict[str, list[list[float]]]:
    """Ground-truth boxes per case, ``[x, y, width, height]`` in 512 px coordinates."""
    boxes = defaultdict(list)
    with (DATA / "boxes_512.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            boxes[r["case_id"]].append([float(r[k]) for k in ("x", "y", "width", "height")])
    return boxes


def read_png(case_id: str) -> np.ndarray:
    with Image.open(DATA / "png512" / f"{case_id}.png") as im:
        return np.asarray(im, dtype=np.uint8)


# ── Transforms ──────────────────────────────────────────────────────────────


class HorizontalFlip:
    """Left-right flip with probability ``p``. Exact: no interpolation, no box is lost."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator):
        if rng.uniform() < self.p:
            W = image.shape[1]
            image = image[:, ::-1].copy()
            boxes = boxes.copy()
            boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        return image, boxes


class Affine:
    """Random affine transform about the image centre, applied to every image (p = 1).

    Drawn uniformly: rotation angle in ``±angle_deg``, isotropic scale factor in
    ``1 ± scale``, translation in ``±translate_px`` on each axis. A point ``p = (x, y)`` of
    the input goes to ``c + s·R(θ)·(p − c) + t`` with ``c = (W/2, H/2)`` and
    ``R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]`` (the y axis points down, so a positive
    angle turns the image clockwise on screen).

    - Image: bilinear interpolation; what comes from outside the input is black (0).
    - Boxes: the same map is applied to the 4 corners; the new box is the axis-aligned
      envelope of the transformed corners (for a non-zero angle it contains the rotated
      box and is larger than it, as in torchvision's ``RandomAffine``), then clipped to the
      image. A box whose clipped width or height is below ``MIN_BOX_SIDE`` (it has left
      the image) is removed. The other boxes keep their order.
    """

    def __init__(self, angle_deg: float = 15.0, translate_px: float = 25.6, scale: float = 0.05):
        self.angle_deg = angle_deg
        self.translate_px = translate_px
        self.scale = scale

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float, float]:
        """``(angle_deg, tx, ty, s)`` drawn from ``rng``."""
        angle = rng.uniform(-self.angle_deg, self.angle_deg)
        tx = rng.uniform(-self.translate_px, self.translate_px)
        ty = rng.uniform(-self.translate_px, self.translate_px)
        s = 1.0 + rng.uniform(-self.scale, self.scale)
        return angle, tx, ty, s

    def __call__(self, image: np.ndarray, boxes: np.ndarray, rng: np.random.Generator):
        return self.apply(image, boxes, *self.sample(rng))

    @staticmethod
    def apply(image: np.ndarray, boxes: np.ndarray, angle_deg: float, tx: float, ty: float, s: float):
        """Deterministic transform with the given parameters (see the class docstring)."""
        H, W = image.shape
        rad = np.deg2rad(angle_deg)
        forward = s * np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
        centre = np.array([W / 2.0, H / 2.0])
        shift = np.array([tx, ty])

        # scipy samples output[o] = input[M @ o + offset], with o in array indices (row, col)
        # and pixel centres at integer indices: continuous (x, y) = (col + 0.5, row + 0.5).
        # Inverse map in (x, y): p = A·p' + b; in indices: q = A·q' + (A·h + b − h), h = ½.
        inverse = np.linalg.inv(forward)
        b = centre - inverse @ (centre + shift)
        half = np.array([0.5, 0.5])
        offset_xy = inverse @ half + b - half
        swap = np.array([[0.0, 1.0], [1.0, 0.0]])  # (x, y) ↔ (row, col)
        out = ndi.affine_transform(image.astype(np.float32), swap @ inverse @ swap, offset=swap @ offset_xy,
                                   output_shape=(H, W), order=1, mode="constant", cval=0.0)
        image = np.clip(np.round(out), 0, 255).astype(np.uint8)

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes):
            x1, y1, x2, y2 = boxes.T.astype(np.float64)
            corners = np.stack([np.stack([x1, y1], -1), np.stack([x2, y1], -1),
                                np.stack([x1, y2], -1), np.stack([x2, y2], -1)], axis=1)  # (N, 4, 2)
            moved = (corners - centre) @ forward.T + centre + shift
            new = np.concatenate([moved.min(axis=1), moved.max(axis=1)], axis=1)
            new = np.clip(new, 0, [W, H, W, H])
            keep = ((new[:, 2] - new[:, 0]) >= MIN_BOX_SIDE) & ((new[:, 3] - new[:, 1]) >= MIN_BOX_SIDE)
            boxes = new[keep].astype(np.float32)
        return image, boxes


class Compose:
    """Chain several transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image, boxes, rng):
        for t in self.transforms:
            image, boxes = t(image, boxes, rng)
        return image, boxes


# 25.6 px = 5 % of 512 px. Vertical flips are deliberately not used: an upside-down chest
# radiograph does not occur in this data.
TRAIN_TRANSFORM = Compose([HorizontalFlip(), Affine(angle_deg=15.0, translate_px=25.6, scale=0.05)])


# ── Dataset ──────────────────────────────────────────────────────────────────


class DetectionDataset(torch.utils.data.Dataset):
    """Returns ``(image, target)`` in the torchvision detection format.

    The greyscale image is repeated on 3 channels, in [0, 1] (the ImageNet normalisation is
    done inside the torchvision model). Boxes are converted to ``[x1, y1, x2, y2]``.
    ``transform(image_uint8, boxes_xyxy, rng) -> (image_uint8, boxes_xyxy)`` is optional.
    """

    def __init__(self, case_ids, boxes, transform=None, seed=0):
        self.case_ids = list(case_ids)
        self.boxes = boxes
        self.transform = transform
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.case_ids)

    def __getitem__(self, i):
        case = self.case_ids[i]
        image = read_png(case)
        xywh = np.asarray(self.boxes.get(case, []), dtype=np.float32).reshape(-1, 4)
        xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
        if self.transform is not None:
            rng = np.random.default_rng([self.seed, self.epoch, i])
            image, xyxy = self.transform(image, xyxy, rng)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).float().div_(255).expand(3, -1, -1)
        target = {
            "boxes": torch.from_numpy(np.ascontiguousarray(xyxy, dtype=np.float32)).reshape(-1, 4),
            "labels": torch.ones(len(xyxy), dtype=torch.int64),
        }
        return tensor, target, case


def collate(batch):
    images, targets, cases = zip(*batch)
    return list(images), list(targets), list(cases)
