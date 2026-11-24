"""The training losses, MTR's `mtr/utils/loss_utils.py` and its decoder's losses: each decoder
step's trajectory, score and waypoint loss, and the dense future prediction's."""

from __future__ import annotations

import torch
import torch.nn.functional as fn

PARTS = ("reg", "cls", "wp", "dense")  # loss parts logged (dense: the MTR decoder's)


def loss(forward, ip, b, wp, idx, slot_weights):
    """The training loss, with the forward pass and intention points given apart (a
    DistributedDataParallel module has no `.ip`): the loss of every decoder step averaged (as
    MTR), plus the dense future prediction's; parts reg, cls, wp (step means) and dense."""
    out = forward(b, wp, all_steps=True)
    losses = [step_loss(traj, score, b, ip, wp, idx, slot_weights) for traj, score in out["steps"]]
    parts = torch.stack([p for _, p in losses]).mean(0)
    dense = _dense_loss(out["dense"], b["agent_fut"], b["agent_fut_valid"])
    value = torch.stack([v for v, _ in losses]).mean() + dense
    return value, torch.cat((parts, dense.detach()[None]))


def _dense_loss(pred, target, valid):
    """MTR's dense future loss with our regression: smooth L1 on x, y plus L1 on vx, vy, per
    valid frame, averaged over the road users with any (then over the batch)."""
    pred, target = pred.float(), target.float()
    per = fn.smooth_l1_loss(pred[..., :2], target[..., :2], reduction="none").sum(-1)
    per = per + (pred[..., 2:] - target[..., 2:]).abs().sum(-1)  # [B, A, F]
    frames = valid.sum(-1)
    agent = (per * valid).sum(-1) / frames.clamp(min=1)
    has = frames > 0
    return ((agent * has).sum(-1) / has.sum(-1).clamp(min=1)).mean()


def step_loss(traj, score, b, ip, wp, idx, slot_weights):
    """The trajectory loss of one prediction (positive mode: the intention point nearest the
    true endpoint, as MTR): smooth L1 on its path, cross entropy on the scores, the distance of
    its path to the given waypoints."""
    fut = b["fut"][..., :2]
    positive = ((fut[:, -1, None, :] - ip[b["typ"].long()]) ** 2).sum(-1).argmin(1)
    best = traj[torch.arange(len(positive)), positive]
    reg = fn.smooth_l1_loss(best, fut, reduction="none").sum(-1).mean()
    cls = fn.cross_entropy(score, positive)
    has = idx >= 0
    at = torch.gather(best, 1, idx.clamp(min=0)[..., None].expand(-1, -1, 2))
    w = torch.tensor(slot_weights, device=wp.device)[None]
    way = ((at - wp[..., :2]).abs().sum(-1) * has * w).sum() / has.sum().clamp(min=1)
    return reg + cls + way, torch.stack([reg.detach(), cls.detach(), way.detach()])
