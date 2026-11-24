"""MTR's dense future prediction and motion decoder (`mtr/models/motion_decoder/mtr_decoder.py`)
for the trajectory generator (`model.RouteModel`), with OnCue's own inputs kept: the route and the
waypoints form a third "condition" source next to MTR's road users and map, and each mode's output
is its control points (the spline bases of `motion_utils`), not MTR's Gaussian mixture.

  - dense future prediction (`DenseFuture`): every road user's future (x, y, vx, vy per frame)
    from its encoded feature and position, encoded back and fused into its feature (MTR; an
    auxiliary loss);
  - decoder (`MtrDecoder`): 64 static intention queries (their sine embedding through a BatchNorm
    MLP), content starting at 0; 6 steps, each with a decoder layer per source
    (`transformer_decoder_layer`), the outputs fused with the center's feature by a BatchNorm
    MLP, a score and a trajectory head per step (all steps supervised), the query position moved
    to the step's predicted endpoint; map attention limited to the 256 segments nearest the point
    30 m ahead and the 128 nearest the query's predicted trajectory (MTR's local movement
    refinement). Added: a learned "no condition" token;
  - `attention_logit_max`: the largest attention logit of each kind of decoder attention, logged
    in training.
"""

from __future__ import annotations

import torch
from torch import nn

from ...utils.motion_utils import start_spline_basis
from ..utils.common_layers import BnMlp, mlp3
from ..utils.transformer.position_encoding_utils import sine_position
from ..utils.transformer.transformer_decoder_layer import DecoderLayer


class DenseFuture(nn.Module):
    """MTR's dense future prediction: every road user's future (x, y, vx, vy per frame, relative
    to its last position) from its feature and position, encoded and fused back into its
    feature. Returns (features, predicted futures [B, A, F, 4] in the center's frame)."""

    def __init__(self, d: int, future: int, mask: bool = True):
        super().__init__()
        self.future, self.mask = future, mask
        self.pos = mlp3(2, d)
        self.head = BnMlp(2 * d, [d, d, future * 4])
        self.encode = mlp3(4 * future, d)
        self.fuse = mlp3(2 * d, d)

    def forward(self, feat, pos, valid):
        x = torch.cat((self.pos(pos), feat), -1)
        on = valid if self.mask else None  # None: every slot (the implementation before 10-23)
        pred = self.head(x, on).view(*feat.shape[:2], self.future, 4)
        pred = torch.cat((pred[..., :2] + pos[:, :, None], pred[..., 2:]), -1)
        fused = self.fuse(torch.cat((feat, self.encode(pred.flatten(2))), -1))
        return fused * valid[..., None], pred


class MtrDecoder(nn.Module):
    """MTR's motion decoder with a third source (route and waypoints). `forward` returns, per
    step, (trajectories [B, K, F, 2] via the spline basis, scores [B, K])."""

    def __init__(self, cfg, enc_width: int, nctrl: int, basis: torch.Tensor):
        super().__init__()
        d, dm, heads, steps = cfg.d_model, 256, cfg.heads, cfg.decoder_layers
        self.d, self.dm, self.steps, self.nctrl = d, dm, steps, nctrl
        self.register_buffer("basis", basis, persistent=False)
        self.in_center = nn.Sequential(nn.Linear(enc_width, d), nn.ReLU(), nn.Linear(d, d))
        self.in_agent = nn.Sequential(nn.Linear(enc_width, d), nn.ReLU(), nn.Linear(d, d))
        self.in_map = nn.Sequential(nn.Linear(enc_width, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.in_route = nn.Sequential(nn.Linear(enc_width, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.dense = DenseFuture(d, cfg.future, cfg.dense_mask)
        self.query = BnMlp(d, [d, d])  # MTR's intention_query_mlps
        self.wp = nn.Sequential(nn.Linear(6 + dm, dm), nn.ReLU(), nn.Linear(dm, dm))
        self.none = nn.Parameter(torch.zeros(dm))  # the "no condition" token
        self.agent_layers = nn.ModuleList(DecoderLayer(d, heads, i == 0) for i in range(steps))
        self.map_layers = nn.ModuleList(DecoderLayer(dm, heads, i == 0) for i in range(steps))
        self.cond_layers = nn.ModuleList(DecoderLayer(dm, heads, i == 0) for i in range(steps))
        self.map_content = nn.ModuleList(nn.Linear(d, dm) for _ in range(steps))
        self.cond_content = nn.ModuleList(nn.Linear(d, dm) for _ in range(steps))
        self.map_pos, self.cond_pos = nn.Linear(d, dm), nn.Linear(d, dm)
        self.fusion = nn.ModuleList(BnMlp(2 * d + 2 * dm, [d, d]) for _ in range(steps))
        self.cls = nn.ModuleList(BnMlp(d, [d, d, 1]) for _ in range(steps))
        self.reg = nn.ModuleList(BnMlp(d, [d, d, nctrl * 2]) for _ in range(steps))
        self.start = None
        if getattr(cfg, "start_velocity", False):
            basis, start = start_spline_basis(cfg.future)
            self.register_buffer("basis", basis, persistent=False)
            self.register_buffer("start_basis", start, persistent=False)
            # the start velocity of a mode where none is given: zero at first, so a model fine-
            # tuned from one without it begins from a standing start there
            self.start = nn.ModuleList(nn.Linear(d, 2) for _ in range(steps))
            for head in self.start:
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)

    def forward(self, ip, agents, agent_pos, agent_ok, maps, map_pos, map_ok, route, route_pos,
                route_ok, wp, base: int = 256, near: int = 128, v0=None, v0_ok=None):  # fmt: skip
        """ip [B, K, 2] intention points; agents [B, A, E] (the center first), maps [B, P, E]
        (nearest the point 30 m ahead first), route [B, R, E]; *_pos [.., 2] positions, *_ok
        valid; wp [B, W, 6]; v0 [B, 2] the center's start velocity in its frame, v0_ok [B] where
        it is known (with `start_velocity`). Returns (steps: [(trajectories, scores)], dense
        futures)."""
        bsz, k = ip.shape[:2]
        d, dm = self.d, self.dm
        center = self.in_center(agents[:, 0])[:, None].expand(-1, k, -1)
        agent_feat = self.in_agent(agents) * agent_ok[..., None]
        agent_feat, dense = self.dense(agent_feat, agent_pos, agent_ok)
        map_feat = self.in_map(maps) * map_ok[..., None]
        present = wp[..., 5] > 0.5
        wtok = self.wp(torch.cat((wp, sine_position(wp[..., :2], dm)), -1))
        cond = torch.cat(
            (self.in_route(route), wtok, self.none.expand(bsz, 1, -1).to(wtok.dtype)), 1
        )
        cond_pos = torch.cat((route_pos, wp[..., :2], wp.new_zeros(bsz, 1, 2)), 1)
        cond_ok = torch.cat((route_ok, present, present.new_ones(bsz, 1)), 1)
        agent_pe, map_pe = sine_position(agent_pos, d), sine_position(map_pos, dm)
        cond_pe = sine_position(cond_pos, dm)
        query_pos = self.query(sine_position(ip, d))
        content = query_pos.new_zeros(bsz, k, d)
        center_xy, path = ip, ip[:, :, None]  # the query position, its predicted path
        base_ok = map_ok.clone()
        base_ok[:, base:] = False  # MTR's base region: the segments nearest the point ahead
        steps = []
        for i in range(self.steps):
            a = self.agent_layers[i](
                content, query_pos, sine_position(center_xy, d), agent_feat, agent_pe,
                agent_ok[:, None],
            )  # fmt: skip
            gap = (path[:, :, None] - map_pos[:, None, :, None]).norm(dim=-1).amin(-1)
            gap = gap.masked_fill(~map_ok[:, None], float("inf"))  # [B, K, P]
            nearest = gap.topk(min(near, gap.shape[-1]), dim=-1, largest=False).indices
            allowed = base_ok[:, None].expand(-1, k, -1).clone()
            allowed.scatter_(-1, nearest, True)
            allowed &= map_ok[:, None]
            allowed[..., 0] |= ~allowed.any(-1)  # never an empty row (no map at all)
            m = self.map_layers[i](
                self.map_content[i](content), self.map_pos(query_pos),
                sine_position(center_xy, dm), map_feat, map_pe, allowed,
            )  # fmt: skip
            c = self.cond_layers[i](
                self.cond_content[i](content), self.cond_pos(query_pos),
                sine_position(center_xy, dm), cond, cond_pe, cond_ok[:, None],
            )  # fmt: skip
            content = self.fusion[i](torch.cat((center, a, m, c), -1))
            score = self.cls[i](content).squeeze(-1)
            raw = self.reg[i](content).view(bsz, k, self.nctrl, 2)
            ctrl = torch.cat((raw.new_zeros(bsz, k, 1, 2), raw), 2)
            traj = torch.einsum("fc,bkcd->bkfd", self.basis.to(ctrl.dtype), ctrl)
            if self.start is not None:  # the start velocity: given where known, else predicted
                d0 = self.start[i](content)  # [B, K, 2]
                if v0 is not None:
                    given = v0.to(d0.dtype)[:, None].expand_as(d0)
                    d0 = torch.where(v0_ok[:, None, None], given, d0)
                traj = traj + self.start_basis.to(d0.dtype)[None, None, :, None] * d0[:, :, None]
            steps.append((traj, score))
            center_xy = traj[:, :, -1]  # MTR: the next step queries around this endpoint
            path = traj[:, :, 9::10]  # one point per second for the map collection
        return steps, dense


def attention_logit_max(model, b: dict, wp) -> dict:
    """Largest |attention logit| per kind of decoder attention (QK-norm bounds them), on the
    given batch (eager, no gradient): logged with the losses as a warning sign."""
    layers = [m for m in model.modules() if isinstance(m, DecoderLayer)]
    for m in layers:
        m.record = []
    with torch.no_grad():
        model(b, wp)
    out = {}
    for name, group in (("agent", "agent_layers"), ("map", "map_layers"), ("cond", "cond_layers")):
        values = [v for m in getattr(model.mtr, group) for v in m.record]
        out[name] = max(values) if values else 0.0
    for m in layers:
        m.record = None
    return out
