"""Arm naming. An arm is a model family paired with a query strategy.

Predictions carry ``model_version = "<family>+<strategy>@<round>"`` so Label
Studio records which arm made each one (PLAN.md, Label Studio integration).
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_VERSION = re.compile(r"^(?P<family>[a-z0-9_-]+)\+(?P<strategy>[a-z0-9_-]+)@(?P<round>\d+)$")


@dataclass(frozen=True)
class Arm:
    family: str
    strategy: str

    def __post_init__(self) -> None:
        for part in (self.family, self.strategy):
            if not _NAME.match(part):
                raise ValueError(f"invalid arm component {part!r}: use lowercase letters, digits, - and _")

    @property
    def name(self) -> str:
        return f"{self.family}+{self.strategy}"

    def model_version(self, round_: int) -> str:
        return f"{self.name}@{round_}"

    @classmethod
    def parse(cls, name: str) -> Arm:
        family, _, strategy = name.partition("+")
        return cls(family, strategy)


def parse_model_version(version: str) -> tuple[Arm, int]:
    """Inverse of ``Arm.model_version``."""
    m = _VERSION.match(version)
    if not m:
        raise ValueError(f"not a Maestro model_version: {version!r}")
    return Arm(m["family"], m["strategy"]), int(m["round"])


def grid(families: list[str], strategies: list[str]) -> list[Arm]:
    """Every family paired with every strategy."""
    return [Arm(f, s) for f, s in itertools.product(families, strategies)]
