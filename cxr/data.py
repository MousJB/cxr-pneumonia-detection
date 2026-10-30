"""Reading the prepared data (``scripts/prepare_rsna.py``) for training and inference."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


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
