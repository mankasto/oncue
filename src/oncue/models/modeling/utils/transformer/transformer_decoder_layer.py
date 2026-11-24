"""The motion decoder's layer, MTR's `TransformerDecoderLayer`
(`mtr/models/utils/transformer/transformer_decoder_layer.py`): self-attention, conditional
cross-attention with content and sine position concatenated per head, feed-forward, post-norm.
Added: query and key LayerNorm in every attention (QK-norm, `multi_head_attention`), which bounds
the attention logits, whose growth collapsed the earlier decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as fn
from torch import nn

from .multi_head_attention import attend


class DecoderLayer(nn.Module):
    """MTR's TransformerDecoderLayer (conditional DETR; post-norm, ReLU feed-forward 4x) with
    global attention and QK-norm."""

    def __init__(self, d: int, heads: int, first: bool, dropout: float = 0.1):
        super().__init__()
        self.heads, self.p, self.first = heads, dropout, first
        lin = lambda: nn.Linear(d, d)  # noqa: E731
        self.sa_qc, self.sa_qp, self.sa_kc, self.sa_kp, self.sa_v = (lin() for _ in range(5))
        self.sa_out = lin()
        self.ca_qc, self.ca_kc, self.ca_kp, self.ca_v = (lin() for _ in range(4))
        if first:  # DETR: the query position joins the content in the first layer only
            self.ca_qp = lin()
        self.ca_qsine, self.ca_out = lin(), lin()
        hd = d // heads
        self.sa_nq, self.sa_nk = nn.LayerNorm(hd), nn.LayerNorm(hd)
        self.ca_nq, self.ca_nk = nn.LayerNorm(2 * hd), nn.LayerNorm(2 * hd)
        self.ff1, self.ff2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.record = None  # a list: monitoring (`attention_logit_max`)

    def forward(self, tgt, query_pos, query_sine, mem, mem_pos, allowed):
        """tgt, query_pos, query_sine [B, Q, d]; mem, mem_pos [B, M, d]; allowed [B, Q or 1, M]."""
        p, train = self.p, self.training
        bq, nq, d = tgt.shape
        everyone = torch.ones(bq, 1, nq, dtype=torch.bool, device=tgt.device)
        q = self.sa_qc(tgt) + self.sa_qp(query_pos)
        k = self.sa_kc(tgt) + self.sa_kp(query_pos)
        a = attend(
            q, k, self.sa_v(tgt), everyone, self.heads, self.sa_nq, self.sa_nk, p, train,
            self.record,
        )  # fmt: skip
        tgt = self.norm1(tgt + fn.dropout(self.sa_out(a), p, train))
        qc, kc, v, kp = self.ca_qc(tgt), self.ca_kc(mem), self.ca_v(mem), self.ca_kp(mem_pos)
        if self.first:
            qc, kc = qc + self.ca_qp(query_pos), kc + kp
        h, hd = self.heads, d // self.heads
        qs = self.ca_qsine(query_sine)
        q = torch.cat((qc.view(bq, nq, h, hd), qs.view(bq, nq, h, hd)), -1).view(bq, nq, 2 * d)
        m = mem.shape[1]
        k = torch.cat((kc.view(bq, m, h, hd), kp.view(bq, m, h, hd)), -1).view(bq, m, 2 * d)
        a = attend(q, k, v, allowed, h, self.ca_nq, self.ca_nk, p, train, self.record)
        tgt = self.norm2(tgt + fn.dropout(self.ca_out(a), p, train))
        y = self.ff2(fn.dropout(fn.relu(self.ff1(tgt)), p, train))
        return self.norm3(tgt + fn.dropout(y, p, train))
