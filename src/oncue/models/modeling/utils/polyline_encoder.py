"""MTR's PointNet polyline encoder (`mtr/models/utils/polyline_encoder.py`), for map segments,
route chunks and road users' frames."""

from __future__ import annotations

import torch
from torch import nn

from .common_layers import bn_mlp, run


class PolylineEncoder(nn.Module):
    """MTR's PointNetPolylineEncoder (map: 64 wide, 5 layers, 3 before the pooled feature is
    appended): MLPs whose BatchNorm counts the valid points only, invalid points 0, max-pooled
    per segment, an output MLP to `out` (0 for an empty segment). Computed on every point (fixed
    shapes); equal to MTR's module, which selects the valid points."""

    def __init__(self, c_in: int, out: int, hidden: int = 64, layers: int = 5, pre: int = 3):
        super().__init__()
        self.hidden = hidden
        self.pre = bn_mlp(c_in, [hidden] * pre)
        self.mlps = bn_mlp(2 * hidden, [hidden] * (layers - pre))
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out))

    def forward(self, x: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
        """x [B, P, Q, C] points, on [B, P, Q] valid points -> [B, P, out]."""
        keep = on[..., None]
        f = run(self.pre, x, on) * keep
        g = torch.cat((f, f.max(2).values[:, :, None].expand_as(f)), -1)
        h = (run(self.mlps, g, on) * keep).max(2).values
        return self.out(h) * on.any(-1)[..., None]
