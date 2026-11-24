"""MTR's sine position embedding (`mtr/models/utils/transformer/position_encoding_utils.py`)."""

from __future__ import annotations

import math

import torch


def sine_position(xy: torch.Tensor, d: int) -> torch.Tensor:
    """MTR's gen_sineembed_for_position: [..., 2] positions (m) -> [..., d] (y half, x half)."""
    half = d // 2
    t = torch.arange(half, dtype=torch.float32, device=xy.device)
    t = 10000 ** (2 * torch.div(t, 2, rounding_mode="floor") / half)
    out = []
    for c in (1, 0):
        a = xy[..., c : c + 1].float() * (2 * math.pi) / t
        out.append(torch.stack((a[..., 0::2].sin(), a[..., 1::2].cos()), -1).flatten(-2))
    return torch.cat(out, -1)
