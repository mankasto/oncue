"""The context encoder's layer, MTR's `TransformerEncoderLayer`
(`mtr/models/utils/transformer/transformer_encoder_layer.py`) and its local attention
(`multi_head_attention_local.py`) in one."""

from __future__ import annotations

import torch
import torch.nn.functional as fn
from torch import nn

from ....ops.attention.gather import gather_plan, gather_rows


class LocalEncoderLayer(nn.Module):
    """MTR's TransformerEncoderLayer with local attention (post-norm, ReLU feed-forward, dropout
    0.1): query and key carry the sine position encoding, the value does not; each input attends
    to its k neighbours only (`knn`), as MTR's attention op: their keys and values gathered,
    softmax and dropout over the k."""

    def __init__(self, d: int, heads: int, ff: int, dropout: float = 0.1):
        super().__init__()
        self.heads, self.p = heads, dropout
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ff1, self.ff2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, pe: torch.Tensor, idx, valid, plan=None) -> torch.Tensor:
        """plan: `gather.gather_plan(idx, T)`, made once for all layers (made here if not given)."""
        b, t, d = x.shape
        h, p = self.heads, (self.p if self.training else 0.0)
        qk = x + pe.to(x.dtype)
        c = d // h
        # products and sums over the k (not tiny matrix products), fused when compiled
        q = self.q(qk).view(b, t, 1, h, c) * c**-0.5
        plan = gather_plan(idx, t) if plan is None else plan
        k_ = idx.shape[-1]  # gather_rows: index_select, its backward without atomic adds
        kn = gather_rows(self.k(qk).reshape(b * t, d), *plan).view(b, t, k_, h, c)
        vn = gather_rows(self.v(x).reshape(b * t, d), *plan).view(b, t, k_, h, c)
        w = (q * kn).sum(-1).masked_fill(~valid[..., None], float("-inf")).softmax(2)
        a = (fn.dropout(w, p, self.training)[..., None] * vn).sum(2)  # [B, T, h, c]
        x = self.norm1(x + fn.dropout(self.o(a.reshape(b, t, d)), p, self.training))
        y = self.ff2(fn.dropout(fn.relu(self.ff1(x)), p, self.training))
        return self.norm2(x + fn.dropout(y, p, self.training))
