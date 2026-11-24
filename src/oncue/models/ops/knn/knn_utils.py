"""The k nearest encoder inputs, in PyTorch, in place of MTR's CUDA op
(`mtr/ops/knn/knn_utils.py`) for the context encoder's local attention."""

from __future__ import annotations

import torch


def knn(pos: torch.Tensor, pad: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(indices [B, T, k], valid [B, T, k]) of each input's k nearest real inputs (itself
    included, as MTR's knn; fewer when there are fewer); a padding input has itself only."""
    dist = torch.cdist(pos.float(), pos.float())
    dist = dist.masked_fill(pad[:, None, :] | pad[:, :, None], float("inf"))
    near = dist.topk(min(k, dist.shape[-1]), dim=-1, largest=False)
    idx, valid = near.indices, near.values.isfinite()
    own = torch.arange(dist.shape[-1], device=pos.device)[None, :]
    idx[..., 0] = torch.where(pad, own, idx[..., 0])  # a padding input: itself
    valid[..., 0] |= pad
    return idx, valid
