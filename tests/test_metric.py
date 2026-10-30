import math

import pytest

from cxr.metric import THRESHOLDS, dataset_score, image_score


def test_thresholds():
    assert list(THRESHOLDS) == [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]


def test_empty_image_is_excluded():
    assert image_score([], [], []) is None
    assert dataset_score({"a": ([], []), "b": ([[0, 0, 10, 10]], [0.9])}, {"a": [], "b": [[0, 0, 10, 10]]}) == (1.0, 1)


def test_false_positive_on_negative_image_scores_zero():
    assert image_score([[0, 0, 10, 10]], [0.9], []) == 0.0


def test_missed_box_scores_zero():
    assert image_score([], [], [[0, 0, 10, 10]]) == 0.0


def test_partial_overlap_counts_only_below_iou():
    # IoU = 50 / 150 = 1/3 < 0.4: miss at every threshold → 1 FP + 1 FN
    assert image_score([[5, 0, 10, 10]], [0.9], [[0, 0, 10, 10]]) == 0.0
    # IoU = 0.6: hit for t < 0.6 (0.40, 0.45, 0.50, 0.55), strict inequality at 0.6
    box = [0, 0, 10, 10]
    pred = [0, 0, 10, 6]
    score = image_score([pred], [0.9], [box])
    assert score == pytest.approx(4 / 8)


def test_duplicate_prediction_is_a_false_positive():
    gt = [[0, 0, 10, 10]]
    good, poor = [0, 0, 10, 10], [0, 0, 10, 7]  # IoU 1.0 and 0.7
    assert image_score([good], [0.9], gt) == 1.0
    # Only one prediction can take the box: 1 TP + 1 FP at every threshold, whichever is
    # the most confident (when poor goes first, it takes the box below 0.70, good above).
    assert image_score([poor, good], [0.1, 0.9], gt) == pytest.approx(0.5)
    assert image_score([poor, good], [0.9, 0.1], gt) == pytest.approx(0.5)


def test_confidence_order_decides_which_prediction_takes_the_box():
    a, b = [0, 0, 10, 10], [4, 0, 10, 10]
    p1 = [1, 0, 10, 10]  # IoU 0.818 with a, 0.538 with b
    p2 = [0, 0, 10, 10]  # IoU 1.0 with a, 0.429 with b
    # p1 first: it takes a; p2 can only take b at t = 0.40 → 1 then 1/3 at 7 thresholds.
    assert image_score([p1, p2], [0.9, 0.1], [a, b]) == pytest.approx((1 + 7 / 3) / 8)
    # p2 first: it takes a; p1 takes b while 0.538 > t (0.40, 0.45, 0.50).
    assert image_score([p1, p2], [0.1, 0.9], [a, b]) == pytest.approx((3 + 5 / 3) / 8)


def test_two_boxes():
    gt = [[0, 0, 10, 10], [50, 50, 10, 10]]
    assert image_score([[0, 0, 10, 10]], [0.5], gt) == pytest.approx(0.5)
    assert image_score([[50, 50, 10, 10], [0, 0, 10, 10]], [0.5, 0.6], gt) == 1.0


def test_dataset_requires_same_images():
    with pytest.raises(ValueError):
        dataset_score({"a": ([], [])}, {"b": []})
    assert math.isnan(dataset_score({"a": ([], [])}, {"a": []})[0])
