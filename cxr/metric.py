"""Competition metric of the RSNA Pneumonia Detection Challenge.

As described by the organisers (Shih et al., Radiology: AI 2019, arXiv:1811.08939):

- for each image and each IoU threshold t in {0.40, 0.45, ..., 0.75}, predictions are
  matched one-to-one to ground-truth boxes **in descending order of confidence**; a
  prediction is a hit if its IoU with a still unmatched ground-truth box is > t;
- the image score at t is TP / (TP + FP + FN); the image score is the mean over the
  thresholds;
- the dataset score is the mean of the image scores over the images that have **at least
  one ground-truth or predicted box**. An image with neither is excluded; an image with
  predictions but no ground truth scores 0.

Boxes are ``(x, y, width, height)`` in pixels.
"""

from collections.abc import Mapping, Sequence

import numpy as np

THRESHOLDS = np.round(np.arange(0.40, 0.76, 0.05), 2)

Box = Sequence[float]


def iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """IoU between every predicted box (rows) and every ground-truth box (columns)."""
    px1, py1 = pred[:, 0:1], pred[:, 1:2]
    px2, py2 = px1 + pred[:, 2:3], py1 + pred[:, 3:4]
    gx1, gy1 = gt[:, 0], gt[:, 1]
    gx2, gy2 = gx1 + gt[:, 2], gy1 + gt[:, 3]
    iw = np.clip(np.minimum(px2, gx2) - np.maximum(px1, gx1), 0, None)
    ih = np.clip(np.minimum(py2, gy2) - np.maximum(py1, gy1), 0, None)
    inter = iw * ih
    union = pred[:, 2:3] * pred[:, 3:4] + gt[:, 2] * gt[:, 3] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def image_score(pred_boxes: Sequence[Box], pred_scores: Sequence[float], gt_boxes: Sequence[Box]) -> float | None:
    """Score of one image, or None if it has neither ground truth nor prediction."""
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    if n_pred == 0 and n_gt == 0:
        return None
    if n_pred == 0 or n_gt == 0:
        return 0.0
    order = np.argsort(-np.asarray(pred_scores, dtype=float), kind="stable")
    ious = iou_matrix(np.asarray(pred_boxes, dtype=float)[order], np.asarray(gt_boxes, dtype=float))
    total = 0.0
    for t in THRESHOLDS:
        matched = np.zeros(n_gt, dtype=bool)
        tp = 0
        for row in ious:
            candidates = np.where(~matched & (row > t), row, -1.0)
            j = int(np.argmax(candidates))
            if candidates[j] > t:
                matched[j] = True
                tp += 1
        total += tp / (n_pred + n_gt - tp)  # TP + FP + FN
    return total / len(THRESHOLDS)


def dataset_score(
    predictions: Mapping[str, tuple[Sequence[Box], Sequence[float]]],
    ground_truth: Mapping[str, Sequence[Box]],
) -> tuple[float, int]:
    """Mean image score and number of images counted. Both mappings cover the same images."""
    if set(predictions) != set(ground_truth):
        raise ValueError("predictions and ground truth must cover the same images")
    scores = [s for case, (boxes, conf) in predictions.items()
              if (s := image_score(boxes, conf, ground_truth[case])) is not None]
    return (float(np.mean(scores)) if scores else float("nan")), len(scores)
