"""Row gather for the local attention: each input's k neighbours' keys and values
taken from a [N, d] tensor by row index, as `index_select`, with a backward that adds no values
atomically. `index_select`'s backward (`index_add`) adds every gathered row's gradient into its
source row atomically: under bf16 on our training GPU 6.7 ms per call, 12 calls per training step (6
layers, keys and values), 29% of the step's GPU time. Here the backward sums each source row's
gradients itself: the neighbour indices are sorted once per forward (all layers share them), and
a Triton kernel reads, per source row, the rows that gathered it and adds them in float32, so the
gradient is also deterministic and summed more precisely than bf16 atomic adds.

  plan = gather_plan(idx, t)        # idx [B, T, k], indices into each batch's T rows
  rows = gather_rows(x, *plan)      # x [B*T, d] -> [B*T*k, d]
"""

from __future__ import annotations

import functools

import torch


def gather_plan(idx: torch.Tensor, t: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(flat, order, start) for gathering rows of a [B*T, d] tensor at idx [B, T, k]: flat, the
    row of every (batch, input, neighbour) [B*T*k]; order, the positions of flat sorted by row
    (stable); row r was gathered at positions order[start[r]:start[r + 1]]."""
    b = idx.shape[0]
    flat = (idx + t * torch.arange(b, device=idx.device)[:, None, None]).flatten()
    rows, order = flat.sort(stable=True)
    start = torch.searchsorted(rows, torch.arange(b * t + 1, device=idx.device))
    return flat, order, start


class _GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, flat, order, start):
        ctx.save_for_backward(order, start)
        return x.index_select(0, flat)

    @staticmethod
    def backward(ctx, grad):
        order, start = ctx.saved_tensors
        return row_sum(grad.contiguous(), order, start), None, None, None


def gather_rows(x, flat, order, start) -> torch.Tensor:
    """x.index_select(0, flat) whose backward sums each row's gradients without atomic adds."""
    return _GatherRows.apply(x, flat, order, start)


@torch.library.custom_op("oncue::row_sum", mutates_args=())
def row_sum(g: torch.Tensor, order: torch.Tensor, start: torch.Tensor) -> torch.Tensor:
    """out[r] = sum of g[order[start[r]:start[r + 1]]], added in float32, g's dtype out."""
    n = start.shape[0] - 1
    if g.is_cuda:
        out = torch.empty(n, g.shape[1], dtype=g.dtype, device=g.device)
        if n:
            block_d = min(256, 1 << (g.shape[1] - 1).bit_length())
            with torch.cuda.device(g.device):
                # one warp, 8 gathering rows at a time: fastest of 64 settings on our training GPU
                # (0.31 ms at the training shape, 3.1 TB/s read; 4 warps: 2.8 ms)
                _row_sum_kernel()[(n, -(-g.shape[1] // block_d))](
                    g, order, start, out, g.shape[1], BLOCK_K=8, BLOCK_D=block_d, num_warps=1
                )
        return out
    dest = torch.repeat_interleave(torch.arange(n, device=g.device), start.diff())
    out = torch.zeros(n, g.shape[1], dtype=torch.float32, device=g.device)
    return out.index_add_(0, dest, g.index_select(0, order).float()).to(g.dtype)


@row_sum.register_fake
def _(g, order, start):
    return g.new_empty(start.shape[0] - 1, g.shape[1])


@functools.cache
def _row_sum_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(g, order, start, out, d, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr):
        r = tl.program_id(0)
        cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        col_ok = cols < d
        lo = tl.load(start + r)
        hi = tl.load(start + r + 1)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for s in range(lo, hi, BLOCK_K):  # BLOCK_K gathering rows at a time
            at = s + tl.arange(0, BLOCK_K)
            src = tl.load(order + at, mask=at < hi, other=0)
            tile = tl.load(
                g + src[:, None] * d + cols[None, :],
                mask=(at < hi)[:, None] & col_ok[None, :],
                other=0.0,
            )
            acc += tl.sum(tile.to(tl.float32), 0)
        tl.store(out + r * d + cols, acc.to(out.dtype.element_ty), mask=col_ok)

    return kernel
