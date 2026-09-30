"""Convert between Maestro instances and Label Studio BrushLabels results.

This is the only module that knows Label Studio's brush RLE format. The
encoding itself comes from the Label Studio SDK so it always matches the UI.
"""

from __future__ import annotations

import uuid
from typing import Any

import numpy as np
from label_studio_sdk.converter import brush

from maestro_common.instances import DEFAULT_LABEL, Instance


def mask_to_rle(mask: np.ndarray) -> list[int]:
    """Encode a 2D boolean mask as Label Studio brush RLE."""
    return brush.mask2rle((np.asarray(mask, dtype=bool) * 255).astype(np.uint8))


def rle_to_mask(rle: list[int], height: int, width: int) -> np.ndarray:
    """Decode Label Studio brush RLE into a 2D boolean mask.

    Brush RLE stores RGBA pixels; strokes drawn in the UI set the alpha
    channel, so that channel decides membership.
    """
    flat = brush.decode_rle(rle)
    return flat.reshape(height, width, 4)[:, :, 3] > 0


def to_ls_results(
    instances: list[Instance],
    *,
    width: int,
    height: int,
    from_name: str,
    to_name: str,
) -> list[dict[str, Any]]:
    """Build the ``result`` list of a Label Studio prediction, one region per instance."""
    results = []
    for inst in instances:
        if inst.mask.shape != (height, width):
            raise ValueError(f"mask shape {inst.mask.shape} does not match {(height, width)}")
        results.append(
            {
                "id": uuid.uuid4().hex[:10],
                "from_name": from_name,
                "to_name": to_name,
                "type": "brushlabels",
                "original_width": width,
                "original_height": height,
                "image_rotation": 0,
                "score": float(inst.score),
                "value": {
                    "format": "rle",
                    "rle": mask_to_rle(inst.mask),
                    "brushlabels": [inst.label],
                },
            }
        )
    return results


def from_ls_results(
    results: list[dict[str, Any]],
    *,
    from_name: str | None = None,
) -> list[Instance]:
    """Read the brush regions of an annotation or prediction ``result`` list.

    Regions of other types (or other control tags, when ``from_name`` is
    given) are ignored. Each brush region becomes one instance.
    """
    instances = []
    for region in results:
        if region.get("type") != "brushlabels":
            continue
        if from_name is not None and region.get("from_name") != from_name:
            continue
        value = region["value"]
        if value.get("format", "rle") != "rle":
            raise ValueError(f"unsupported brush format {value.get('format')!r}")
        mask = rle_to_mask(value["rle"], region["original_height"], region["original_width"])
        labels = value.get("brushlabels") or [DEFAULT_LABEL]
        instances.append(Instance(mask=mask, score=float(region.get("score", 1.0)), label=labels[0]))
    return instances
