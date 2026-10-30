"""Augmentation transforms: the boxes must follow the image.

Every test draws a white rectangle on a black background, transforms the image and its
box, and compares the returned box with the bounding box of the white pixels actually
found in the transformed image. Images are not square (W ≠ H) so that a swap of the x and
y axes cannot go unnoticed.
"""

import numpy as np
import pytest
import torch

from cxr import data
from cxr.data import MIN_BOX_SIDE, TRAIN_TRANSFORM, Affine, Compose, DetectionDataset, HorizontalFlip

H, W = 96, 128


def rectangle_image(x1, y1, x2, y2, h=H, w=W):
    """Black image with the pixels of the box ``[x1, y1, x2, y2]`` (integers) in white."""
    image = np.zeros((h, w), dtype=np.uint8)
    image[y1:y2, x1:x2] = 255
    return image, np.array([[x1, y1, x2, y2]], dtype=np.float32)


def white_extent(image, level=128):
    """``[x1, y1, x2, y2]`` (box convention) of the pixels ≥ ``level``, or None."""
    rows, cols = np.nonzero(image >= level)
    if len(rows) == 0:
        return None
    return np.array([cols.min(), rows.min(), cols.max() + 1, rows.max() + 1], dtype=np.float64)


class FixedUniform:
    """Stands in for a Generator: ``uniform`` returns the given values in turn."""

    def __init__(self, *values):
        self.values = list(values)

    def uniform(self, low=0.0, high=1.0):
        return low + self.values.pop(0) * (high - low)


# ── Horizontal flip ──────────────────────────────────────────────────────────


def test_flip_moves_box_with_rectangle():
    image, boxes = rectangle_image(10, 20, 40, 50)
    out, out_boxes = HorizontalFlip(p=1.0)(image, boxes, np.random.default_rng(0))
    np.testing.assert_array_equal(out, image[:, ::-1])
    np.testing.assert_array_equal(out_boxes, [[W - 40, 20, W - 10, 50]])
    np.testing.assert_array_equal(white_extent(out), out_boxes[0])


def test_flip_probability_zero_and_draw_decides():
    image, boxes = rectangle_image(10, 20, 40, 50)
    out, out_boxes = HorizontalFlip(p=0.0)(image, boxes, np.random.default_rng(0))
    np.testing.assert_array_equal(out, image)
    np.testing.assert_array_equal(out_boxes, boxes)
    _, flipped = HorizontalFlip()(image, boxes, FixedUniform(0.49))
    _, kept = HorizontalFlip()(image, boxes, FixedUniform(0.51))
    np.testing.assert_array_equal(flipped, [[W - 40, 20, W - 10, 50]])
    np.testing.assert_array_equal(kept, boxes)


def test_flip_does_not_modify_inputs():
    image, boxes = rectangle_image(10, 20, 40, 50)
    image0, boxes0 = image.copy(), boxes.copy()
    HorizontalFlip(p=1.0)(image, boxes, np.random.default_rng(0))
    np.testing.assert_array_equal(image, image0)
    np.testing.assert_array_equal(boxes, boxes0)


# ── Affine: known transforms ─────────────────────────────────────────────────


def test_identity_is_exact():
    image, boxes = rectangle_image(30, 20, 60, 40)
    image[5, 7] = 99  # non-uniform content
    out, out_boxes = Affine.apply(image, boxes, 0.0, 0.0, 0.0, 1.0)
    np.testing.assert_array_equal(out, image)
    np.testing.assert_array_equal(out_boxes, boxes)


@pytest.mark.parametrize("tx, ty", [(10, -7), (-25, 13), (0, 30)])
def test_integer_translation_is_exact(tx, ty):
    image, boxes = rectangle_image(30, 20, 60, 40)
    out, out_boxes = Affine.apply(image, boxes, 0.0, tx, ty, 1.0)
    expected, _ = rectangle_image(30 + tx, 20 + ty, 60 + tx, 40 + ty)
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_allclose(out_boxes, [[30 + tx, 20 + ty, 60 + tx, 40 + ty]], atol=1e-4)


def test_translation_direction_reported_by_reviewer():
    """Case from the audit of the previous implementation: box ended at [121, 76, 321, 396]."""
    image, boxes = rectangle_image(300, 100, 420, 200, h=512, w=512)
    out, out_boxes = Affine.apply(image, boxes, 0.0, 24.0, -21.0, 1.0)
    np.testing.assert_allclose(out_boxes, [[324, 79, 444, 179]], atol=1e-4)
    np.testing.assert_allclose(white_extent(out), out_boxes[0], atol=0)


def test_rotation_by_90_degrees_matches_rot90():
    """Square image: a quarter turn about the centre is an exact pixel permutation."""
    image, boxes = rectangle_image(10, 20, 40, 30, h=64, w=64)
    out, out_boxes = Affine.apply(image, boxes, 90.0, 0.0, 0.0, 1.0)
    # Positive angle = clockwise on screen (y down): np.rot90 with k=-1.
    np.testing.assert_array_equal(out, np.rot90(image, k=-1))
    # Clockwise quarter turn of a W×W image: (x, y) -> (W − y, x).
    np.testing.assert_allclose(out_boxes, [[64 - 30, 10, 64 - 20, 40]], atol=1e-4)
    np.testing.assert_array_equal(white_extent(out), out_boxes[0])


@pytest.mark.parametrize("angle", [-15.0, -7.5, 5.0, 15.0])
def test_rotation_envelope_contains_rotated_rectangle(angle):
    image, boxes = rectangle_image(40, 30, 80, 60)
    out, out_boxes = Affine.apply(image, boxes, angle, 0.0, 0.0, 1.0)
    # Envelope of the rotated corners of a filled rectangle = extent of the rotated rectangle.
    np.testing.assert_allclose(white_extent(out), out_boxes[0], atol=1.0)
    # Analytical size of the envelope: w·|cos| + h·|sin| by w·|sin| + h·|cos|.
    c, s = abs(np.cos(np.deg2rad(angle))), abs(np.sin(np.deg2rad(angle)))
    width, height = out_boxes[0, 2] - out_boxes[0, 0], out_boxes[0, 3] - out_boxes[0, 1]
    assert width == pytest.approx(40 * c + 30 * s, abs=1e-3)
    assert height == pytest.approx(40 * s + 30 * c, abs=1e-3)
    # Rotation about the centre: the box centre (60, 45) is moved to c + R(centre_box − c).
    rad = np.deg2rad(angle)
    dx, dy = 60 - W / 2, 45 - H / 2
    expected_centre = (W / 2 + np.cos(rad) * dx - np.sin(rad) * dy, H / 2 + np.sin(rad) * dx + np.cos(rad) * dy)
    np.testing.assert_allclose(out_boxes[0, :2] + [width / 2, height / 2], expected_centre, atol=1e-3)


@pytest.mark.parametrize("s", [0.95, 1.05, 1.5])
def test_scaling_about_centre(s):
    image, boxes = rectangle_image(40, 30, 80, 60)
    out, out_boxes = Affine.apply(image, boxes, 0.0, 0.0, 0.0, s)
    expected = [W / 2 + s * (40 - W / 2), H / 2 + s * (30 - H / 2), W / 2 + s * (80 - W / 2), H / 2 + s * (60 - H / 2)]
    np.testing.assert_allclose(out_boxes, [expected], atol=1e-4)
    np.testing.assert_allclose(white_extent(out), out_boxes[0], atol=1.0)


def test_scale_is_isotropic():
    """One scale factor for both axes: the aspect ratio of an unrotated box is kept."""
    image, boxes = rectangle_image(40, 30, 80, 60)
    for seed in range(20):
        angle, tx, ty, s = Affine(angle_deg=0.0).sample(np.random.default_rng(seed))
        _, out_boxes = Affine.apply(image, boxes, angle, tx, ty, s)
        w, h = out_boxes[0, 2] - out_boxes[0, 0], out_boxes[0, 3] - out_boxes[0, 1]
        assert w / h == pytest.approx(40 / 30, rel=1e-5)


def test_combined_transform_order():
    """Rotation + scale about the centre, then translation."""
    image, boxes = rectangle_image(40, 30, 80, 60)
    out, out_boxes = Affine.apply(image, boxes, 10.0, 6.0, -4.0, 1.04)
    np.testing.assert_allclose(white_extent(out), out_boxes[0], atol=1.0)
    rad = np.deg2rad(10.0)
    corners = np.array([[40, 30], [80, 30], [40, 60], [80, 60]], dtype=float) - [W / 2, H / 2]
    moved = corners @ (1.04 * np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])).T
    moved += [W / 2 + 6.0, H / 2 - 4.0]
    np.testing.assert_allclose(out_boxes[0], [*moved.min(0), *moved.max(0)], atol=1e-3)


@pytest.mark.parametrize("seed", range(25))
def test_random_draws_keep_alignment(seed):
    """Random parameters as in training: returned box = extent of the white pixels."""
    image, boxes = rectangle_image(40, 30, 80, 60)
    affine = Affine(angle_deg=15.0, translate_px=0.05 * W, scale=0.05)
    out, out_boxes = affine(image, boxes, np.random.default_rng(seed))
    assert out_boxes.shape == (1, 4)
    np.testing.assert_allclose(white_extent(out), out_boxes[0], atol=1.0)


def test_train_transform_keeps_alignment():
    """The full training pipeline (flip + affine) on a 512 px image."""
    image, boxes = rectangle_image(300, 100, 420, 200, h=512, w=512)
    for seed in range(10):
        out, out_boxes = TRAIN_TRANSFORM(image, boxes, np.random.default_rng(seed))
        np.testing.assert_allclose(white_extent(out), out_boxes[0], atol=1.0)


def test_same_rng_gives_same_result():
    image, boxes = rectangle_image(40, 30, 80, 60)
    a = TRAIN_TRANSFORM(image, boxes, np.random.default_rng([0, 3, 17]))
    b = TRAIN_TRANSFORM(image, boxes, np.random.default_rng([0, 3, 17]))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


# ── Affine: borders and degenerate boxes ─────────────────────────────────────


def test_box_partly_outside_is_clipped():
    image, boxes = rectangle_image(100, 30, 120, 60)
    out, out_boxes = Affine.apply(image, boxes, 0.0, 18.0, 0.0, 1.0)  # x2 → 138 > W
    np.testing.assert_allclose(out_boxes, [[118, 30, W, 60]], atol=1e-4)
    np.testing.assert_array_equal(white_extent(out), out_boxes[0])


def test_box_leaving_the_image_is_removed():
    image, boxes = rectangle_image(100, 30, 120, 60)
    out, out_boxes = Affine.apply(image, boxes, 0.0, 40.0, 0.0, 1.0)  # x1 → 140 > W
    assert out_boxes.shape == (0, 4)
    assert white_extent(out) is None


def test_sliver_below_minimum_side_is_removed():
    image, boxes = rectangle_image(100, 30, 120, 60)
    out_boxes = Affine.apply(image, boxes, 0.0, 27.5, 0.0, 1.0)[1]  # 0.5 px left inside
    assert out_boxes.shape == (0, 4)
    kept = Affine.apply(image, boxes, 0.0, 28.0 - MIN_BOX_SIDE, 0.0, 1.0)[1]  # exactly 1 px
    np.testing.assert_allclose(kept, [[W - MIN_BOX_SIDE, 30, W, 60]], atol=1e-4)


def test_removal_keeps_order_of_remaining_boxes():
    image = np.zeros((H, W), dtype=np.uint8)
    boxes = np.array([[5, 5, 20, 20], [110, 40, 125, 50], [30, 60, 50, 80]], dtype=np.float32)
    _, out_boxes = Affine.apply(image, boxes, 0.0, -30.0, 0.0, 1.0)  # the first box leaves on the left
    np.testing.assert_allclose(out_boxes, [[80, 40, 95, 50], [0, 60, 20, 80]], atol=1e-4)


def test_no_boxes():
    image = np.full((H, W), 128, dtype=np.uint8)
    out, out_boxes = Affine()(image, np.empty((0, 4), dtype=np.float32), np.random.default_rng(0))
    assert out.shape == (H, W) and out.dtype == np.uint8
    assert out_boxes.shape == (0, 4) and out_boxes.dtype == np.float32


def test_compose_applies_in_order():
    image, boxes = rectangle_image(10, 20, 40, 50)
    out, out_boxes = Compose([HorizontalFlip(p=1.0), lambda im, b, rng: Affine.apply(im, b, 0.0, 5.0, 0.0, 1.0)])(
        image, boxes, np.random.default_rng(0))
    np.testing.assert_allclose(out_boxes, [[W - 40 + 5, 20, W - 10 + 5, 50]], atol=1e-4)
    np.testing.assert_array_equal(white_extent(out), out_boxes[0])


# ── Dataset: image, boxes and labels stay consistent ─────────────────────────


@pytest.fixture
def fake_png(monkeypatch):
    image, _ = rectangle_image(100, 30, 120, 60)
    monkeypatch.setattr(data, "read_png", lambda case_id: image.copy())
    return image


def test_dataset_converts_xywh_to_xyxy(fake_png):
    ds = DetectionDataset(["a", "b"], {"a": [[100, 30, 20, 30]]})
    tensor, target, case = ds[0]
    assert case == "a" and tensor.shape == (3, H, W)
    assert torch.allclose(tensor[0], torch.from_numpy(fake_png).float() / 255)
    assert torch.equal(target["boxes"], torch.tensor([[100.0, 30.0, 120.0, 60.0]]))
    assert torch.equal(target["labels"], torch.tensor([1]))
    _, empty, _ = ds[1]
    assert empty["boxes"].shape == (0, 4) and empty["labels"].shape == (0,)


def test_dataset_labels_follow_removed_boxes(fake_png):
    def push_right(image, boxes, rng):
        return Affine.apply(image, boxes, 0.0, 15.0, 0.0, 1.0)  # the 2nd box leaves the image

    ds = DetectionDataset(["a"], {"a": [[100, 30, 20, 30], [125, 70, 2, 10]]}, transform=push_right)
    tensor, target, _ = ds[0]
    assert target["boxes"].shape == (1, 4)
    assert torch.equal(target["labels"], torch.tensor([1]))
    assert torch.allclose(target["boxes"], torch.tensor([[115.0, 30.0, W, 60.0]]))
    extent = white_extent((tensor[0].numpy() * 255).round().astype(np.uint8))
    np.testing.assert_allclose(extent, target["boxes"][0].numpy(), atol=0)


def test_dataset_augmentation_depends_on_seed_epoch_index(fake_png):
    ds = DetectionDataset(["a", "a"], {"a": [[100, 30, 20, 30]]}, transform=TRAIN_TRANSFORM, seed=1)
    first = ds[0][1]["boxes"]
    assert torch.equal(first, ds[0][1]["boxes"])  # same (seed, epoch, index): same draw
    assert not torch.equal(first, ds[1][1]["boxes"])  # other index
    ds.epoch = 1
    assert not torch.equal(first, ds[0][1]["boxes"])  # other epoch
