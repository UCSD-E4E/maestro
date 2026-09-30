import numpy as np
import pytest

from maestro_common.instances import Instance
from maestro_common.ls_brush import from_ls_results, mask_to_rle, rle_to_mask, to_ls_results


def disk(h, w, cy, cx, r):
    yy, xx = np.mgrid[:h, :w]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (240, 320)])
def test_rle_round_trip(shape):
    rng = np.random.default_rng(0)
    mask = rng.random(shape) > 0.6
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), *shape), mask)


def test_empty_and_full_masks_round_trip():
    for mask in (np.zeros((5, 9), bool), np.ones((5, 9), bool)):
        assert np.array_equal(rle_to_mask(mask_to_rle(mask), 5, 9), mask)


def test_results_round_trip_keeps_instances_scores_and_labels():
    h, w = 60, 80
    instances = [Instance(disk(h, w, 20, 20, 8), 0.9), Instance(disk(h, w, 40, 60, 10), 0.3, "fish")]
    results = to_ls_results(instances, width=w, height=h, from_name="tag", to_name="image")

    assert [r["type"] for r in results] == ["brushlabels", "brushlabels"]
    assert results[0]["value"]["brushlabels"] == ["fish"]
    assert results[0]["original_width"] == w and results[0]["original_height"] == h

    back = from_ls_results(results)
    assert len(back) == 2
    for a, b in zip(instances, back):
        assert np.array_equal(a.mask, b.mask)
        assert b.score == pytest.approx(a.score)
        assert b.label == a.label


def test_from_ls_results_skips_other_types_and_tags():
    h, w = 10, 10
    results = to_ls_results([Instance(disk(h, w, 5, 5, 3))], width=w, height=h, from_name="tag", to_name="image")
    results.append({"type": "rectanglelabels", "from_name": "box", "value": {}})
    assert len(from_ls_results(results)) == 1
    assert from_ls_results(results, from_name="other") == []


def test_mask_shape_must_match_image():
    with pytest.raises(ValueError):
        to_ls_results([Instance(np.zeros((4, 4), bool))], width=5, height=4, from_name="t", to_name="i")
