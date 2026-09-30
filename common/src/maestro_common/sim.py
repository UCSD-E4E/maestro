"""Offline bandit simulator: the Sherlock Fig. 5 experiment, before any labels exist.

Each model family follows a synthetic learning curve (mean per-image IoU as a
function of labels trained on). A query strategy's efficiency scales how much
each label is worth. Every round the policy plays one arm: its strategy adds
a batch of labels, its family retrains on everything labeled so far, and the
retrained model is scored on the round's audit images (PLAN.md, Bandit and
reward).

The default curves and efficiencies are illustrative, not measured. They
exist to exercise the bandit and tune r, gamma, round size and audit share;
replace them with fitted curves once real rounds exist.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from maestro_common.arms import Arm
from maestro_common.bandit import BetaBandit


@dataclass(frozen=True)
class LearningCurve:
    """Mean IoU after n effective labels: plateau - (plateau - start) * exp(-n / rate)."""

    start: float
    plateau: float
    rate: float

    def __call__(self, n: float) -> float:
        return self.plateau - (self.plateau - self.start) * math.exp(-n / self.rate)


@dataclass(frozen=True)
class SyntheticFamily:
    name: str
    curve: LearningCurve
    trainable: bool = True


class SyntheticEnvironment:
    def __init__(
        self,
        families: list[SyntheticFamily],
        *,
        strategy_efficiency: dict[str, float],
        rng: np.random.Generator,
        concentration: float = 20.0,
    ) -> None:
        self.families = {f.name: f for f in families}
        self.strategy_efficiency = strategy_efficiency
        self.rng = rng
        self.concentration = concentration  # higher = less per-image noise
        self.labels = 0
        self.effective_labels = 0.0
        self._trained_on = {name: 0.0 for name in self.families}

    def arms(self) -> list[Arm]:
        return [Arm(f, s) for f, s in itertools.product(self.families, self.strategy_efficiency)]

    def play(self, arm: Arm, batch_size: int) -> None:
        """The arm's strategy picks a batch; its family retrains on all labels so far."""
        self.labels += batch_size
        self.effective_labels += batch_size * self.strategy_efficiency[arm.strategy]
        if self.families[arm.family].trainable:
            self._trained_on[arm.family] = self.effective_labels

    def quality(self, family: str) -> float:
        return self.families[family].curve(self._trained_on[family])

    def best_quality(self) -> float:
        """Mean IoU of the best model: what labelers see as pre-annotations."""
        return max(self.quality(name) for name in self.families)

    def audit_rewards(self, family: str, n: int) -> np.ndarray:
        """Per-image IoUs on n audit images, Beta-distributed around the family's quality."""
        mean = min(max(self.quality(family), 1e-3), 1 - 1e-3)
        return self.rng.beta(mean * self.concentration, (1 - mean) * self.concentration, size=n)


def default_environment(rng: np.random.Generator) -> SyntheticEnvironment:
    """Illustrative curves: zero-shot is flat, YOLO-seg learns fast, Mask R-CNN
    learns slowly but ends higher, so the best arm changes over time."""
    return SyntheticEnvironment(
        [
            SyntheticFamily("zero-shot", LearningCurve(0.55, 0.55, 1.0), trainable=False),
            SyntheticFamily("yolo-seg", LearningCurve(0.25, 0.75, 400.0)),
            SyntheticFamily("mask-rcnn", LearningCurve(0.15, 0.85, 1200.0)),
        ],
        strategy_efficiency={"uncertainty": 1.4, "coreset": 1.2, "random": 1.0},
        rng=rng,
    )


@dataclass(frozen=True)
class SimConfig:
    rounds: int = 40
    batch_size: int = 50
    audit_fraction: float = 0.15
    r: float = 1.0
    gamma: float = 0.9


@dataclass
class RoundRecord:
    round: int
    labels: int
    arm: str
    mean_reward: float
    best_quality: float
    posterior_means: dict[str, float] = field(default_factory=dict)


def run(
    env: SyntheticEnvironment,
    config: SimConfig,
    rng: np.random.Generator,
    *,
    fixed_arm: str | None = None,
) -> list[RoundRecord]:
    """Play ``config.rounds`` rounds with the bandit, or always ``fixed_arm``."""
    bandit = BetaBandit([a.name for a in env.arms()], r=config.r, gamma=config.gamma)
    n_audit = max(1, round(config.batch_size * config.audit_fraction))
    history = []
    for t in range(1, config.rounds + 1):
        arm_name = fixed_arm or bandit.select(rng)[0]
        arm = Arm.parse(arm_name)
        env.play(arm, config.batch_size)
        rewards = env.audit_rewards(arm.family, n_audit)
        for x in rewards:
            bandit.update(arm_name, float(x))
        bandit.discount()
        history.append(
            RoundRecord(t, env.labels, arm_name, float(rewards.mean()), env.best_quality(), bandit.means())
        )
    return history


@dataclass
class PolicyResult:
    labels: np.ndarray  # labels after each round
    curve: np.ndarray  # mean best quality after each round, over seeds
    target: float
    posterior_trace: list[dict[str, float]] = field(default_factory=list)  # first seed only

    @property
    def final_quality(self) -> float:
        return float(self.curve[-1])

    @property
    def labels_to_target(self) -> int | None:
        """Labels needed before the mean best quality first reaches the target."""
        hit = np.nonzero(self.curve >= self.target)[0]
        return int(self.labels[hit[0]]) if len(hit) else None


def compare(config: SimConfig, *, seeds: int = 20, target: float = 0.7) -> dict[str, PolicyResult]:
    """The bandit against every fixed arm, each run on the same seeds."""
    arm_names = [a.name for a in default_environment(np.random.default_rng(0)).arms()]
    results = {}
    for policy in ["bandit", *arm_names]:
        curves, trace = [], []
        for seed in range(seeds):
            env = default_environment(np.random.default_rng(seed))
            history = run(env, config, np.random.default_rng(10_000 + seed), fixed_arm=None if policy == "bandit" else policy)
            curves.append([h.best_quality for h in history])
            if seed == 0:
                trace = [h.posterior_means for h in history]
        labels = np.array([config.batch_size * t for t in range(1, config.rounds + 1)])
        results[policy] = PolicyResult(labels, np.mean(curves, axis=0), target, trace)
    return results
