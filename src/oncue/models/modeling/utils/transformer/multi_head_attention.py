"""Multi-head attention with QK-norm for the decoder layers, in place of MTR's
`MultiheadAttention` (`mtr/models/utils/transformer/multi_head_attention.py`)."""

from __future__ import annotations

import torch.nn.functional as fn


def attend(q, k, v, allowed, heads: int, norm_q, norm_k, p: float, training: bool, record=None):
    """Multi-head attention of q [B, Q, Dq] over k [B, M, Dq], v [B, M, Dv]; allowed [B, Q, M]
    or [B, 1, M] (True: attended); q and k layer-normed per head (QK-norm). `record`: a list
    the largest |logit| over the attended keys is appended to (monitoring)."""
    b, nq, dq = q.shape
    m, dv = k.shape[1], v.shape[-1]
    q = norm_q(q.view(b, nq, heads, dq // heads)).transpose(1, 2)
    k = norm_k(k.view(b, m, heads, dq // heads)).transpose(1, 2)
    v = v.view(b, m, heads, dv // heads).transpose(1, 2)
    if record is not None:
        logit = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        record.append(float(logit.masked_fill(~allowed[:, None], 0).abs().max()))
    out = fn.scaled_dot_product_attention(
        q, k, v, attn_mask=allowed[:, None], dropout_p=p if training else 0.0
    )
    return out.transpose(1, 2).reshape(b, nq, dv)
