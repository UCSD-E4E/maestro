"""Query strategies, and turning their choices into Label Studio scores.

A strategy ranks the unlabeled tasks, most useful to label first. The round's
queue is the top of that ranking with random audit tasks mixed in (PLAN.md,
Bandit and reward). Label Studio Enterprise's uncertainty sampling serves
the lowest prediction score first, so ``queue_scores`` gives each task a score
equal to its queue position divided by the pool size.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np

from maestro_common.instances import Instance

STRATEGIES = ("uncertainty", "coreset", "random")


def instance_uncertainty(score: float) -> float:
    """1 at confidence 0.5, 0 at confidence 0 or 1."""
    return 1.0 - abs(2.0 * score - 1.0)


def uncertainty_value(instances: list[Instance]) -> float:
    """An image is as uncertain as its least certain candidate detection.

    Models should return candidates below their display threshold too;
    those borderline fish are what make an image worth labeling. An image
    with no candidates scores 0.
    """
    return max((instance_uncertainty(inst.score) for inst in instances), default=0.0)


def rank_uncertainty(task_ids: Sequence[Hashable], predictions: dict[Hashable, list[Instance]]) -> list:
    values = {tid: uncertainty_value(predictions.get(tid, [])) for tid in task_ids}
    return sorted(task_ids, key=lambda tid: -values[tid])


def rank_coreset(
    task_ids: Sequence[Hashable],
    embeddings: np.ndarray,
    labeled_embeddings: np.ndarray,
) -> list:
    """Greedy k-center order: each next task is the one farthest from everything
    already labeled or already ranked ahead of it."""
    n = len(task_ids)
    if n == 0:
        return []
    emb = np.asarray(embeddings, dtype=np.float64)
    dist = np.full(n, np.inf)
    # Squared distance to the nearest labeled embedding, in chunks to bound memory.
    lab = np.asarray(labeled_embeddings, dtype=np.float64).reshape(-1, emb.shape[1])
    for start in range(0, len(lab), 1024):
        chunk = lab[start : start + 1024]
        d2 = (emb**2).sum(1)[:, None] + (chunk**2).sum(1)[None, :] - 2.0 * emb @ chunk.T
        dist = np.minimum(dist, d2.min(axis=1))
    order: list[int] = []
    remaining = np.ones(n, dtype=bool)
    for _ in range(n):
        i = int(np.argmax(np.where(remaining, dist, -np.inf)))
        order.append(i)
        remaining[i] = False
        dist = np.minimum(dist, ((emb - emb[i]) ** 2).sum(-1))
    return [task_ids[i] for i in order]


def rank_random(task_ids: Sequence[Hashable], rng: np.random.Generator) -> list:
    return [task_ids[i] for i in rng.permutation(len(task_ids))]


def build_queue(
    ranked: Sequence[Hashable],
    batch_size: int,
    audit_fraction: float,
    rng: np.random.Generator,
) -> tuple[list, set]:
    """The next batch to label: top of the ranking plus uniform-random audit tasks.

    Audit tasks are spread evenly through the batch so labelers reach them
    at a steady rate. Returns (queue, audit task ids).
    """
    if not 0.0 <= audit_fraction <= 1.0:
        raise ValueError("audit_fraction must be in [0, 1]")
    batch_size = min(batch_size, len(ranked))
    n_audit = int(round(batch_size * audit_fraction))
    audit = [ranked[i] for i in rng.choice(len(ranked), size=n_audit, replace=False)]
    audit_set = set(audit)
    picked = [tid for tid in ranked if tid not in audit_set][: batch_size - n_audit]
    queue = list(picked)
    for k, tid in enumerate(audit):
        queue.insert(round((k + 1) * len(queue) / (n_audit + 1)) + k, tid)
    return queue, audit_set


def queue_scores(queue: Sequence[Hashable], ranked: Sequence[Hashable]) -> dict:
    """Prediction scores that make Label Studio serve ``queue`` first, in order,
    then the rest of the ranking. Scores lie in [0, 1); lower is served sooner."""
    order = list(queue) + [tid for tid in ranked if tid not in set(queue)]
    return {tid: i / len(order) for i, tid in enumerate(order)}
