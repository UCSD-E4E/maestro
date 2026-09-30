import numpy as np
import pytest

from maestro_common.arms import Arm, grid, parse_model_version
from maestro_common.instances import Instance
from maestro_common.strategies import (
    build_queue,
    queue_scores,
    rank_coreset,
    rank_random,
    rank_uncertainty,
    uncertainty_value,
)


def test_model_version_round_trip():
    arm = Arm("yolo-seg", "coreset")
    assert arm.model_version(7) == "yolo-seg+coreset@7"
    assert parse_model_version("yolo-seg+coreset@7") == (arm, 7)
    assert Arm.parse(arm.name) == arm


def test_invalid_names_and_versions_are_rejected():
    with pytest.raises(ValueError):
        Arm("YOLO", "coreset")
    with pytest.raises(ValueError):
        parse_model_version("something-else")


def test_grid_is_the_full_product():
    assert len(grid(["a", "b", "c"], ["x", "y", "z"])) == 9


def test_uncertainty_peaks_at_half_confidence():
    m = np.zeros((2, 2), bool)
    assert uncertainty_value([Instance(m, 0.5)]) == 1.0
    assert uncertainty_value([Instance(m, 0.99), Instance(m, 0.6)]) == pytest.approx(0.8)
    assert uncertainty_value([]) == 0.0


def test_rank_uncertainty_puts_borderline_images_first():
    m = np.zeros((2, 2), bool)
    preds = {1: [Instance(m, 0.95)], 2: [Instance(m, 0.5)], 3: []}
    assert rank_uncertainty([1, 2, 3], preds) == [2, 1, 3]


def test_coreset_picks_the_far_cluster_first_then_spreads_out():
    labeled = np.array([[0.0, 0.0]])
    emb = np.array([[0.1, 0.0], [10.0, 10.0], [10.1, 10.0], [5.0, 5.0]])
    order = rank_coreset(["near", "far", "far2", "mid"], emb, labeled)
    assert order[0] in {"far", "far2"}
    assert order[1] == "mid"
    assert order[-1] in {"near", "far", "far2"}


def test_coreset_without_labels_still_orders_everything():
    emb = np.random.default_rng(0).random((20, 4))
    assert sorted(rank_coreset(list(range(20)), emb, np.zeros((0, 4)))) == list(range(20))


def test_queue_has_batch_size_with_audits_mixed_in():
    rng = np.random.default_rng(0)
    ranked = list(range(100))
    queue, audit = build_queue(ranked, batch_size=20, audit_fraction=0.15, rng=rng)
    assert len(queue) == len(set(queue)) == 20
    assert len(audit) == 3 and audit <= set(queue)
    non_audit = [t for t in queue if t not in audit]
    assert non_audit == [t for t in ranked if t not in audit][:17]
    assert queue[0] not in audit and queue[-1] not in audit  # spread through the batch


def test_queue_scores_serve_queue_first_in_order():
    ranked = rank_random(list("abcdef"), np.random.default_rng(0))
    queue = ["e", "a"]
    scores = queue_scores(queue, ranked)
    served = sorted(scores, key=scores.get)
    assert served[:2] == queue
    assert set(served) == set("abcdef")
    assert all(0 <= s < 1 for s in scores.values())
