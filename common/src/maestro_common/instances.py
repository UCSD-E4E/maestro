"""The task-agnostic unit Maestro passes between models, rewards and Label Studio."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_LABEL = "fish"


@dataclass
class Instance:
    """One predicted or annotated object: a boolean mask plus a confidence.

    Annotations from people have ``score=1.0``.
    """

    mask: np.ndarray  # bool, shape (height, width)
    score: float = 1.0
    label: str = DEFAULT_LABEL

    def __post_init__(self) -> None:
        self.mask = np.asarray(self.mask, dtype=bool)
        if self.mask.ndim != 2:
            raise ValueError(f"mask must be 2D, got shape {self.mask.shape}")


def stack_masks(instances: list[Instance], shape: tuple[int, int]) -> np.ndarray:
    """Masks as one (n, height, width) bool array; (0, h, w) when empty."""
    if not instances:
        return np.zeros((0, *shape), dtype=bool)
    masks = np.stack([inst.mask for inst in instances])
    if masks.shape[1:] != tuple(shape):
        raise ValueError(f"mask shape {masks.shape[1:]} does not match image shape {shape}")
    return masks
