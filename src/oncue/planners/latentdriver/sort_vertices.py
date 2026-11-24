"""PyTorch port of LatentDriver's CUDA vertex sort (``sort_vert_kernel.cu``, commit 721bd6e).

``install`` lets the original rotated-box IoU import it, so fine-tuning needs no CUDA kernel.
"""

from __future__ import annotations

import sys
import types

import torch

MAX_NUM_VERT_IDX = 9  # ``sort_vert_kernel.cu``: at most 8 vertices, with the first repeated
INTERSECTION_OFFSET = 8  # four + four box corners come first, edge intersections after them
MIN_POLYGON_VERTICES = 3


def install() -> None:
    """Expose ``sort_v`` at ``src.ops.sort_vertices.sort_vert`` before the IoU is imported."""
    ops, package = types.ModuleType("src.ops"), types.ModuleType("src.ops.sort_vertices")
    module = types.ModuleType("src.ops.sort_vertices.sort_vert")
    module.sort_v = sort_v
    sys.modules.setdefault("src.ops", ops)
    sys.modules.setdefault("src.ops.sort_vertices", package)
    sys.modules["src.ops.sort_vertices.sort_vert"] = module


def sort_v(vertices: torch.Tensor, mask: torch.Tensor, num_valid: torch.Tensor) -> torch.Tensor:
    """Return ``[B, N, 9]`` vertex indices of each intersection polygon in angular order.

    ``vertices [B, N, 24, 2]`` are centred on the valid ones' mean; ``mask [B, N, 24]`` marks
    the valid ones; ``num_valid [B, N]`` counts them. Like the kernel, it assumes at most 8 valid
    vertices and has no gradient.
    """
    angles = torch.atan2(vertices[..., 1], vertices[..., 0]).masked_fill(~mask, float("inf"))
    order = angles.argsort(dim=-1)[..., :MAX_NUM_VERT_IDX]
    # An invalid edge intersection: the IoU zeroes those, so pointing at one adds nothing.
    pad = (~mask[..., INTERSECTION_OFFSET:]).int().argmax(dim=-1, keepdim=True)
    pad = (pad + INTERSECTION_OFFSET).expand_as(order)
    position = torch.arange(MAX_NUM_VERT_IDX, device=vertices.device)
    count = num_valid.long().unsqueeze(-1)
    indices = torch.where(position < count, order, pad)
    # Close the polygon by repeating its first vertex; fewer than 3 vertices give zero area.
    indices = torch.where(position == count, order[..., :1].expand_as(order), indices)
    return torch.where(count < MIN_POLYGON_VERTICES, pad, indices)
