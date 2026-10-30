"""Evaluation of PROTOCOL.md §7.3, §8, §9.1."""

import numpy as np
import pytest

from cxr.evaluation import (
    image_stats,
    match_boxes,
    non_inferiority,
    paired_bootstrap,
    ratios,
    selection_rule,
    weighted_auroc,
)


def test_match_boxes_greedy_by_score():
    gt = [[0, 0, 10, 10], [100, 100, 10, 10]]
    pred = [[0, 0, 10, 10], [1, 1, 10, 10], [200, 200, 5, 5]]
    gt_idx, ious = match_boxes(pred, [0.5, 0.9, 0.7], gt)
    assert gt_idx == [0]  # the 0.9 box takes gt 0 (IoU 0.68); the exact one comes later and finds none
    assert ious[0] == pytest.approx(81 / 119)


def test_image_stats_and_ratios():
    truth = {"pos": [[0, 0, 120, 120], [200, 200, 50, 50]], "neg": [], "miss": [[0, 0, 100, 100]]}
    preds = {"pos": (np.array([[0, 0, 120, 120], [300, 300, 20, 20]], float), np.array([0.8, 0.6])),
             "neg": (np.array([[0, 0, 10, 10]], float), np.array([0.2])),
             "miss": (np.zeros((0, 4)), np.zeros(0))}
    stats = image_stats(preds, truth, tau=0.5)
    assert stats["neg"]["rsna_count"] == 0  # its only box is below τ: neither truth nor prediction
    assert stats["pos"]["small_gt"] == 1 and stats["pos"]["small_matched"] == 0  # 50×50 < 11,503 px²
    from cxr.evaluation import FIELDS
    r = ratios(np.array([[s[f] for f in FIELDS] for s in stats.values()], float).sum(0))
    assert r["sensitivity"] == 0.5 and r["specificity"] == 1.0 and r["precision_image"] == 1.0
    assert r["box_recall"] == pytest.approx(1 / 3) and r["box_precision"] == 0.5
    assert r["mean_iou_matched"] == 1.0
    # rsna: "pos" = mean over 8 thresholds of 1/(2+2-1) = 1/3; "miss" = 0; "neg" excluded
    assert r["rsna_metric"] == pytest.approx((1 / 3 + 0) / 2)


def test_weighted_auroc():
    assert weighted_auroc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert weighted_auroc([0.5, 0.5], [1, 0]) == 0.5
    assert weighted_auroc([0.9, 0.1, 0.5], [1, 0, 0]) == 1.0
    assert weighted_auroc([0.3, 0.1, 0.5], [1, 0, 0]) == 0.5
    # weight 2 on the negative scored 0.5 = duplicating it
    assert weighted_auroc([0.3, 0.1, 0.5], [1, 0, 0], [1, 1, 2]) == pytest.approx(1 / 3)


def test_paired_bootstrap_identical_models_give_zero_difference():
    rng = np.random.default_rng(0)
    truth, preds, patients = {}, {}, {}
    for i in range(60):
        pos = i % 3 == 0
        truth[f"c{i}"] = [[10, 10, 50, 50]] if pos else []
        score = rng.uniform(0.5, 1) if pos else rng.uniform(0, 0.6)
        preds[f"c{i}"] = (np.array([[12, 12, 50, 50]], float), np.array([score]))
        patients[f"c{i}"] = f"p{i // 2}"
    stats = image_stats(preds, truth, tau=0.5)
    out = paired_bootstrap({"A": stats, "B": stats}, patients, resamples=200, seed=1)
    d = out["differences"]["B - A"]
    assert all(v["value"] == 0 and v["ci"] == [0.0, 0.0] for v in d.values())
    a = out["models"]["A"]["sensitivity"]
    assert a["ci"][0] <= a["value"] <= a["ci"][1]
    assert out["patients"] == 30 and out["images"] == 60


def test_selection_rule():
    good = {"rsna_metric": {"value": -0.005, "ci": [-0.015, 0.004]},
            "sensitivity": {"value": 0.0, "ci": [-0.03, 0.03]},
            "small_box_recall": {"value": -0.02, "ci": [-0.1, 0.05]}}
    assert selection_rule(good)["selected"] == "B"
    for key, bad in (("rsna_metric", {"value": -0.01, "ci": [-0.025, 0.0]}),
                     ("sensitivity", {"value": -0.02, "ci": [-0.06, 0.01]}),
                     ("small_box_recall", {"value": -0.06, "ci": [-0.2, 0.0]})):
        assert selection_rule({**good, key: bad})["selected"] == "A"


def test_non_inferiority_reading():
    assert non_inferiority([-0.01, 0.02], 0.02) == "SATISFIED"
    assert non_inferiority([-0.05, -0.03], 0.02) == "NOT_SATISFIED"
    assert non_inferiority([-0.03, 0.01], 0.02) == "INCONCLUSIVE"
