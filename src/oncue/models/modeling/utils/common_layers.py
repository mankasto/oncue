"""MLP builders shared by the model's encoders and decoder, MTR's
`mtr/models/utils/common_layers.py` (`build_mlps`), with BatchNorm over the valid positions only."""

from __future__ import annotations

import torch
from torch import nn


def bn_mlp(c_in: int, widths: list[int]) -> nn.Sequential:
    """MTR's build_mlps: Linear (no bias), BatchNorm, ReLU per layer."""
    layers = []
    for w in widths:
        layers += [nn.Linear(c_in, w, bias=False), nn.BatchNorm1d(w), nn.ReLU()]
        c_in = w
    return nn.Sequential(*layers)


def _masked_bn(bn: nn.BatchNorm1d, x: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
    """`bn` over the valid points `on` only (as MTR's BatchNorm on the selected points), applied to
    every point; fixed shapes (no selection), so the model compiles (`torch.compile`)."""
    if not bn.training:
        return (x - bn.running_mean) / torch.sqrt(bn.running_var + bn.eps) * bn.weight + bn.bias
    w = on[..., None].float()
    n = w.sum()
    some = n > 0  # a batch without any valid point: zero output, running statistics unchanged
    xf = x.float()
    mean = (xf * w).sum(tuple(range(x.dim() - 1))) / n.clamp(min=1)
    var = ((xf - mean) ** 2 * w).sum(tuple(range(x.dim() - 1))) / n.clamp(min=1)
    with torch.no_grad():  # running statistics as BatchNorm1d (unbiased variance)
        # a non-finite batch leaves them as they are (skipping its step later
        # cannot undo a NaN written here, which inference would then read); its output stays
        # non-finite, so the step is still skipped
        ok = some & torch.isfinite(mean).all() & torch.isfinite(var).all()
        if bn.momentum is None:  # the cumulative average (BatchNorm1d's momentum None)
            m = ok.float() / (bn.num_batches_tracked.float() + 1)
        else:
            m = bn.momentum * ok.float()
        new_mean = (1 - m) * bn.running_mean + m * mean
        new_var = (1 - m) * bn.running_var + m * var * n / (n - 1).clamp(min=1)
        bn.running_mean.copy_(torch.where(ok, new_mean, bn.running_mean))
        bn.running_var.copy_(torch.where(ok, new_var, bn.running_var))
        bn.num_batches_tracked.add_(ok.long())
    y = (xf - mean) / torch.sqrt(var + bn.eps) * bn.weight + bn.bias
    return torch.where(some, y, torch.zeros_like(y)).to(x.dtype)


def run(seq: nn.Sequential, x: torch.Tensor, on: torch.Tensor) -> torch.Tensor:
    for layer in seq:
        x = _masked_bn(layer, x, on) if isinstance(layer, nn.BatchNorm1d) else layer(x)
    return x


class BnMlp(nn.Module):
    """MTR's build_mlps(..., ret_before_act=True): (Linear, BatchNorm, ReLU) per hidden layer, a
    plain Linear last; on [..., C] (BatchNorm over every leading position)."""

    def __init__(self, c_in: int, widths: list[int]):
        super().__init__()
        self.hidden = bn_mlp(c_in, widths[:-1])
        self.last = nn.Linear(widths[-2] if len(widths) > 1 else c_in, widths[-1])

    def forward(self, x, on=None):
        """`on`: the valid positions (default all); BatchNorm counts only those, as MTR applies
        the MLP to the selected valid ones."""
        if on is None:
            on = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        return self.last(run(self.hidden, x, on))


def mlp3(c_in: int, d: int) -> nn.Sequential:
    """MTR's build_mlps(c_in, [d, d, d], ret_before_act=True, without_norm=True)."""
    return nn.Sequential(nn.Linear(c_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
