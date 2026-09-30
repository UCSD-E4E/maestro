"""Thompson sampling over Beta posteriors, following Sherlock (PLAN.md).

Each arm keeps Beta(alpha, beta). A reward x in [0, 1] (here, per-image IoU)
is applied as a fractional update: alpha += r*x, beta += r*(1 - x), where r is
Sherlock's reshaping factor. ``discount`` pulls every posterior back toward
the prior by gamma so that old evidence fades as the best arm changes.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class ArmPosterior:
    alpha: float
    beta: float
    plays: int = 0  # rounds this arm was selected
    updates: int = 0  # rewards applied

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


class BetaBandit:
    def __init__(
        self,
        arms: list[str],
        *,
        r: float = 1.0,
        gamma: float = 1.0,
        prior: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if not arms:
            raise ValueError("a bandit needs at least one arm")
        if r <= 0:
            raise ValueError("r must be positive")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.r = r
        self.gamma = gamma
        self.prior = prior
        self.arms: dict[str, ArmPosterior] = {}
        self.sync_arms(arms)

    def sync_arms(self, arms: list[str]) -> None:
        """Add arms missing from the state (at the prior) and drop ones no longer configured."""
        self.arms = {
            name: self.arms.get(name, ArmPosterior(*self.prior)) for name in dict.fromkeys(arms)
        }

    def sample(self, rng: np.random.Generator) -> dict[str, float]:
        """One Thompson draw per arm."""
        return {name: float(rng.beta(p.alpha, p.beta)) for name, p in self.arms.items()}

    def select(self, rng: np.random.Generator, n: int = 1) -> list[str]:
        """Draw once from every posterior and return the n arms with the largest draws.

        With n > 1 this is batched Thompson sampling: one round, several
        trainers, distinct arms.
        """
        draws = self.sample(rng)
        chosen = sorted(draws, key=draws.get, reverse=True)[: min(n, len(draws))]
        for name in chosen:
            self.arms[name].plays += 1
        return chosen

    def update(self, arm: str, reward: float) -> None:
        if not 0.0 <= reward <= 1.0:
            raise ValueError(f"reward must be in [0, 1], got {reward}")
        p = self.arms[arm]
        p.alpha += self.r * reward
        p.beta += self.r * (1.0 - reward)
        p.updates += 1

    def discount(self) -> None:
        a0, b0 = self.prior
        for p in self.arms.values():
            p.alpha = a0 + self.gamma * (p.alpha - a0)
            p.beta = b0 + self.gamma * (p.beta - b0)

    def means(self) -> dict[str, float]:
        return {name: p.mean for name, p in self.arms.items()}

    # Persistence: the scheduler rewrites this file after every update.

    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "gamma": self.gamma,
            "prior": list(self.prior),
            "arms": {name: asdict(p) for name, p in self.arms.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> BetaBandit:
        bandit = cls(list(data["arms"]), r=data["r"], gamma=data["gamma"], prior=tuple(data["prior"]))
        bandit.arms = {name: ArmPosterior(**p) for name, p in data["arms"].items()}
        return bandit

    def save(self, path: str | Path) -> None:
        """Write atomically so a crash never leaves a half-written state file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
        with os.fdopen(fd, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path) -> BetaBandit:
        return cls.from_dict(json.loads(Path(path).read_text()))
