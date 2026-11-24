"""The road-user part of MTR's context encoder (`mtr/models/context_encoder/mtr_encoder.py`):
MTR's PointNet over the frames of each road user (BatchNorm, 256 wide, 3 layers), the center
itself one of them (flagged), the frame as a one-hot (`agent_frames`: MTR's 29 attributes).
`model.RouteModel` puts the encoder together (map and route PointNets, local attention layers)."""

from __future__ import annotations

import torch
import torch.nn.functional as fn
from torch import nn

from ..utils.polyline_encoder import PolylineEncoder


class AgentEncoder(nn.Module):
    """MTR's agent polyline encoder: each road user's frames as the points of a PointNet
    (BatchNorm, 256 wide, 3 layers, the pooled feature appended after the first)."""

    def __init__(self, c_in: int, out: int):
        super().__init__()
        self.net = PolylineEncoder(c_in, out, hidden=256, layers=3, pre=1)

    def forward(self, frames, seen):
        """frames [B, A, H, C], seen [B, A, H] -> [B, A, out] (0 for a road user never seen)."""
        return self.net(frames, seen)


def agent_frames(b: dict, history: int):
    """([B, 1 + N, H, C] per-frame inputs, [B, 1 + N, H] seen) of the center (first) and its
    neighbours: x, y, cos, sin, speed, valid, vx, vy, ax, ay, z, type (3), length, width,
    height, is SDC, is the center, the frame (one-hot), as MTR's 29 attributes."""
    hist, nbr = b["hist"].float(), b["nbr"].float()  # [B, H, 11], [B, N, H, 16]
    bsz, n = hist.shape[0], nbr.shape[1]
    dev = hist.device
    typ = fn.one_hot(b["typ"].long().clamp(0, 2), 3).float()
    own_static = torch.cat(
        (typ, b["size"].float(), b["sdc"].float()[:, None], hist.new_ones(bsz, 1)), -1
    )
    nbr_motion = torch.cat((nbr[..., :6], nbr[..., 9:13], nbr[..., 15:16]), -1)
    nbr_typ = fn.one_hot(nbr[..., 6].long().clamp(0, 2), 3).float()
    nbr_static = torch.cat(
        (nbr_typ, nbr[..., 7:9], nbr[..., 13:15], nbr.new_zeros(bsz, n, history, 1)), -1
    )
    motion = torch.cat((hist[:, None], nbr_motion), 1)  # [B, 1 + N, H, 11]
    static = torch.cat((own_static[:, None, None].expand(-1, 1, history, -1), nbr_static), 1)
    time = torch.eye(history, device=dev)[None, None].expand(bsz, 1 + n, -1, -1)
    seen = motion[..., 5] > 0.5
    return torch.cat((motion, static, time), -1) * seen[..., None], seen
