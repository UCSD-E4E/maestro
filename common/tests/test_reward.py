import numpy as np
import pytest

from maestro_common.instances import Instance
from maestro_common.reward import image_iou, iou_matrix, match

H, W = 50, 50


def box(y0, x0, y1, x1):
    m = np.zeros((H, W), bool)
    m[y0:y1, x0:x1] = True
    return Instance(m)


def test_perfect_prediction_scores_one():
    fish = [box(0, 0, 10, 10), box(20, 20, 30, 30)]
    assert image_iou(fish, fish) == pytest.approx(1.0)


def test_empty_image_with_no_predictions_scores_one():
    assert image_iou([], []) == 1.0


def test_false_positive_on_empty_image_scores_zero():
    assert image_iou([box(0, 0, 5, 5)], []) == 0.0


def test_all_missed_scores_zero():
    assert image_iou([], [box(0, 0, 5, 5)]) == 0.0


def test_partial_overlap_is_the_iou():
    # 10x10 box vs the same box shifted 2 rows: intersection 80, union 120.
    assert image_iou([box(2, 0, 12, 10)], [box(0, 0, 10, 10)]) == pytest.approx(80 / 120)


def test_overlap_below_threshold_is_not_a_match():
    # intersection 25, union 175: IoU 0.14 < 0.5.
    assert image_iou([box(5, 5, 15, 15)], [box(0, 0, 10, 10)]) == 0.0


def test_extra_prediction_is_penalized():
    fish = [box(0, 0, 10, 10)]
    assert image_iou(fish + [box(30, 30, 40, 40)], fish) == pytest.approx(0.5)


def test_matching_is_one_to_one_and_optimal():
    gts = [box(0, 0, 10, 10), box(0, 8, 10, 18)]
    preds = [box(0, 1, 10, 11), box(0, 8, 10, 18)]
    pairs = sorted((p, g) for p, g, _ in match(preds, gts))
    assert pairs == [(0, 0), (1, 1)]


def test_iou_matrix_shapes_when_empty():
    assert iou_matrix(np.zeros((0, H, W), bool), np.zeros((3, H, W), bool)).shape == (0, 3)


def test_score_is_order_independent():
    rng = np.random.default_rng(1)
    preds = [Instance(rng.random((H, W)) > 0.5) for _ in range(4)]
    gts = [Instance(rng.random((H, W)) > 0.5) for _ in range(3)]
    assert image_iou(preds, gts, 0.0) == pytest.approx(image_iou(preds[::-1], gts[::-1], 0.0))


def labeled(inst, label):
    return Instance(inst.mask, inst.score, label)


def test_class_aware_matching_rejects_wrong_class():
    fish = labeled(box(0, 0, 10, 10), "fish")
    model = labeled(box(0, 0, 10, 10), "fish model")
    assert image_iou([model], [fish], class_aware=True) == 0.0
    assert image_iou([model], [fish]) == pytest.approx(1.0)  # class-agnostic by default


def test_class_aware_matching_pairs_within_each_class():
    gts = [labeled(box(0, 0, 10, 10), "fish"), labeled(box(0, 12, 10, 22), "fish model")]
    preds = [labeled(box(0, 12, 10, 22), "fish model"), labeled(box(0, 0, 10, 10), "fish")]
    assert sorted((p, g) for p, g, _ in match(preds, gts, class_aware=True)) == [(0, 1), (1, 0)]
    assert image_iou(preds, gts, class_aware=True) == pytest.approx(1.0)


def test_class_aware_prefers_same_class_over_higher_iou():
    # The "fish model" prediction overlaps the fish best, but may only match the model.
    gts = [labeled(box(0, 0, 10, 10), "fish"), labeled(box(0, 2, 10, 12), "fish model")]
    preds = [labeled(box(0, 0, 10, 10), "fish model")]
    ((p, g, iou),) = match(preds, gts, class_aware=True)
    assert (p, g) == (0, 1) and iou == pytest.approx(80 / 120)
