"""Metrics of PROTOCOL.md §7.3 with paired patient bootstrap, selection rule §8, reading of
non-inferiority §9.1.

Predictions: ``{case: (boxes_xywh (N, 4), scores (N,))}`` (every box ≥ 0.01; τ is applied
here). Ground truth: ``{case: [[x, y, w, h], ...]}`` (empty list = no opacity).
"""

import numpy as np

from cxr.metric import image_score, iou_matrix

SMALL_AREA = 11_503.0  # px² at 512 px: smallest third of the training boxes (§7.3)
BOX_IOU = 0.4  # §7.3: box-level match
FIELDS = ("rsna_sum", "rsna_count", "pos", "neg", "tp_img", "tn_img", "pred_pos",
          "gt", "gt_matched", "pred", "pred_matched", "iou_sum", "small_gt", "small_matched")


def match_boxes(pred_boxes, pred_scores, gt_boxes, threshold=BOX_IOU):
    """Predictions in decreasing score, each matched to the unmatched ground-truth box of highest
    IoU if > threshold. Returns ``(matched_gt_indices, ious_of_matches)``."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], []
    order = np.argsort(-np.asarray(pred_scores, dtype=float), kind="stable")
    ious = iou_matrix(np.asarray(pred_boxes, dtype=float)[order], np.asarray(gt_boxes, dtype=float))
    matched = np.zeros(len(gt_boxes), dtype=bool)
    gt_idx, values = [], []
    for row in ious:
        candidates = np.where(~matched & (row > threshold), row, -1.0)
        j = int(np.argmax(candidates))
        if candidates[j] > threshold:
            matched[j] = True
            gt_idx.append(j)
            values.append(float(row[j]))
    return gt_idx, values


def image_stats(predictions, ground_truth, tau: float) -> dict:
    """Per image: the additive counts of ``FIELDS`` and the maximum box score."""
    out = {}
    for case, gt in ground_truth.items():
        boxes, scores = predictions[case]
        keep = scores >= tau
        boxes, scores = boxes[keep], scores[keep]
        gt = np.asarray(gt, dtype=float).reshape(-1, 4)
        positive, predicted = len(gt) > 0, len(boxes) > 0
        s = image_score(boxes, scores, gt)
        gt_idx, ious = match_boxes(boxes, scores, gt)
        small = gt[:, 2] * gt[:, 3] < SMALL_AREA
        out[case] = {
            "rsna_sum": 0.0 if s is None else s, "rsna_count": 0 if s is None else 1,
            "pos": int(positive), "neg": int(not positive),
            "tp_img": int(positive and predicted), "tn_img": int(not positive and not predicted),
            "pred_pos": int(predicted), "gt": len(gt), "gt_matched": len(gt_idx),
            "pred": len(boxes), "pred_matched": len(gt_idx), "iou_sum": float(sum(ious)),
            "small_gt": int(small.sum()), "small_matched": int(small[gt_idx].sum()) if gt_idx else 0,
            "max_score": float(predictions[case][1].max()) if len(predictions[case][1]) else 0.0,
        }
    return out


def ratios(sums: np.ndarray) -> dict:
    """Metrics from summed counts (last axis = ``FIELDS``); works on bootstrap matrices too."""
    f = {name: sums[..., i] for i, name in enumerate(FIELDS)}
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "rsna_metric": f["rsna_sum"] / f["rsna_count"],
            "sensitivity": f["tp_img"] / f["pos"],
            "specificity": f["tn_img"] / f["neg"],
            "precision_image": f["tp_img"] / f["pred_pos"],
            "box_recall": f["gt_matched"] / f["gt"],
            "box_precision": f["pred_matched"] / f["pred"],
            "mean_iou_matched": f["iou_sum"] / f["gt_matched"],
            "small_box_recall": f["small_matched"] / f["small_gt"],
        }


def weighted_auroc(scores, labels, weights=None) -> float:
    """AUROC = P(score of a positive > score of a negative) + ½ P(tie), images weighted."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    w = np.ones(len(scores)) if weights is None else np.asarray(weights, float)
    order = np.argsort(scores, kind="stable")
    s, y, w = scores[order], labels[order], w[order]
    _, start = np.unique(s, return_index=True)
    wp = np.add.reduceat(np.where(y, w, 0.0), start)
    wn = np.add.reduceat(np.where(~y, w, 0.0), start)
    neg_below = np.cumsum(wn) - wn
    total_p, total_n = wp.sum(), wn.sum()
    if total_p == 0 or total_n == 0:
        return float("nan")
    return float((wp * (neg_below + 0.5 * wn)).sum() / (total_p * total_n))


def paired_bootstrap(stats: dict[str, dict], patients: dict[str, str], resamples=2000, seed=20261030,
                     level=0.95) -> dict:
    """Point estimates and percentile CIs for every model and every difference to the first model.

    ``stats = {model: image_stats(...)}``, all over the same images; resampling by patient, the
    same draws for every model (paired).
    """
    models = list(stats)
    cases = sorted(next(iter(stats.values())))
    pid = sorted({patients[c] for c in cases})
    index = {p: i for i, p in enumerate(pid)}
    case_patient = np.array([index[patients[c]] for c in cases])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(pid), size=(resamples, len(pid)))
    counts = np.stack([np.bincount(d, minlength=len(pid)) for d in draws])  # (B, patients)
    weights = counts[:, case_patient]  # (B, images)
    lo, hi = 100 * (1 - level) / 2, 100 * (1 + level) / 2

    per_model, samples = {}, {}
    for m in models:
        table = np.array([[stats[m][c][f] for f in FIELDS] for c in cases], dtype=float)
        point = ratios(table.sum(0))
        boot = ratios(weights @ table)
        labels = table[:, FIELDS.index("pos")] > 0
        max_scores = np.array([stats[m][c]["max_score"] for c in cases])
        point["auroc_image"] = weighted_auroc(max_scores, labels)
        boot["auroc_image"] = np.array([weighted_auroc(max_scores, labels, w) for w in weights])
        samples[m] = boot
        per_model[m] = {k: {"value": float(v), "ci": [float(np.nanpercentile(boot[k], lo)),
                                                      float(np.nanpercentile(boot[k], hi))]}
                        for k, v in point.items()}
    diffs = {}
    ref = models[0]
    for m in models[1:]:
        diffs[f"{m} - {ref}"] = {
            k: {"value": per_model[m][k]["value"] - per_model[ref][k]["value"],
                "ci": [float(np.nanpercentile(samples[m][k] - samples[ref][k], lo)),
                       float(np.nanpercentile(samples[m][k] - samples[ref][k], hi))]}
            for k in per_model[m]}
    return {"models": per_model, "differences": diffs, "resamples": resamples, "seed": seed, "level": level,
            "images": len(cases), "patients": len(pid)}


def selection_rule(diff: dict) -> dict:
    """PROTOCOL.md §8, ``diff`` = differences B − A on validation."""
    c1 = diff["rsna_metric"]["ci"][0] > -0.02
    c2 = diff["sensitivity"]["ci"][0] > -0.05
    c3 = diff["small_box_recall"]["value"] >= -0.05
    return {"condition_1_rsna_lower_bound_gt_-0.02": bool(c1),
            "condition_2_sensitivity_lower_bound_gt_-0.05": bool(c2),
            "condition_3_small_box_recall_diff_ge_-0.05": bool(c3),
            "selected": "B" if (c1 and c2 and c3) else "A"}


def non_inferiority(ci, margin: float) -> str:
    """Reading of §9.1 (same as MedVision QA)."""
    if ci[0] > -margin:
        return "SATISFIED"
    if ci[1] < -margin:
        return "NOT_SATISFIED"
    return "INCONCLUSIVE"
