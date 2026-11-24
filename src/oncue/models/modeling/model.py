"""The trajectory generator of the paper (`RouteModel`, MTR's `MotionTransformer` in
`mtr/models/model.py`), shared by vehicles, pedestrians and cyclists, and its loader
(`load_route_model`). The model puts MTR's context encoder together: the road users' PointNet
(`mtr_encoder`), PointNet map and route encoders, local attention over each input's nearest
inputs (`transformer_encoder_layer`, at the positions `_token_positions` gives); then MTR's dense
future prediction and motion decoder (`mtr_decoder`). It is conditioned on the route
("navigation"), the signal phases and up to `dataset.MAX_WAYPOINTS` timed waypoints; each mode is
NCTRL control points turned into its future positions (`motion_utils`).

Inputs (agent frame, `waymo_dataset.assemble` plus):
  route [R, Q, 3]  the agent's route as polyline chunks (x, y, valid), 1 m spacing, R = 8 chunks of
                   Q = 16 points from the point nearest the agent; vehicles: the lane chain of its
                   route; pedestrians and cyclists: their crossing line
  route_ok         1 if a route is given
  msig [P, 9]      per map segment, its lane's signal phase at the anchor and every 1 s after
                   (0 none or unknown, 1 stop, 2 caution, 3 go; last reported phase held)
"""

from __future__ import annotations

import torch
import torch.nn.functional as fn
from torch import nn

from ..config import ModelConfig, model_config
from ..datasets.routes import SIGNAL_TIMES
from ..datasets.waymo.data_preprocess import feature_config
from ..datasets.waymo.waymo_dataset import N_MAP_TYPES
from ..ops.attention.gather import gather_plan
from ..ops.knn.knn_utils import knn
from ..utils.motion_utils import NCTRL, spline_basis
from .context_encoder.mtr_encoder import AgentEncoder, agent_frames
from .motion_decoder.mtr_decoder import MtrDecoder
from .utils.polyline_encoder import PolylineEncoder
from .utils.transformer.position_encoding_utils import sine_position
from .utils.transformer.transformer_encoder_layer import LocalEncoderLayer


class RouteModel(nn.Module):
    """Parameters and buffers keep the names and order of the checkpoints trained since 10-21
    (`ip`, `basis`, then `pt`, `enc`, `agent_enc`, `rt_enc`, `mtr`)."""

    def __init__(self, cfg: ModelConfig, points, history: int = 11):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("ip", points)
        self.register_buffer("basis", spline_basis(cfg.future))
        e = cfg.encoder_width  # the encoder's width (MTR: 256, its decoder 512)
        # map segments: points with MTR's attributes and their lane's signal phases
        self.pt = PolylineEncoder(2 + N_MAP_TYPES + 6 + SIGNAL_TIMES * 4, e)
        self.enc = nn.ModuleList(
            LocalEncoderLayer(e, cfg.heads, cfg.encoder_ff * e) for _ in range(cfg.encoder_layers)
        )
        self.agent_enc = AgentEncoder(11 + 8 + history, e)
        self.rt_enc = PolylineEncoder(3, e)  # route chunks as polylines (x, y, order)
        self.mtr = MtrDecoder(cfg, e, NCTRL, self.basis)

    def forward(self, b: dict, wp: torch.Tensor, all_steps: bool = False):
        """b: one batch of model inputs (`inputs.track_inputs`); wp [B, 3, 6]: x, y, t / 8 s, speed,
        speed given, present. Returns trajectories [B, K, F, 2] and scores [B, K] (the decoder's
        last step; all_steps: {"steps": [(trajectories, scores)], "dense": futures}). Road users
        through MTR's agent PointNet (the center first), map and route segments through
        PointNets, MTR's local-attention encoder, then `mtr_decoder`."""
        batch, history = b["hist"].shape[0], b["hist"].shape[1]
        frames, seen = agent_frames(b, history)
        agents = self.agent_enc(frames, seen)  # [B, 1 + N, E]
        agent_ok = seen.any(-1)
        nbr = b["nbr"].float()
        mp = b["map"].float()
        kind = mp[..., 2].long()
        sig = fn.one_hot(b["msig"].long().clamp(0, 3), 4).float().flatten(-2)
        feats = [mp[..., :2], fn.one_hot(kind, N_MAP_TYPES).float(), mp[..., 3:9],
                 sig[:, :, None].expand(-1, -1, mp.shape[2], -1)]  # fmt: skip
        maps = self.pt(torch.cat(feats, -1), kind > 0)
        map_ok = kind[..., 0] > 0
        rt = b["route"].float()
        on = rt[..., 2] > 0.5
        order = torch.arange(rt.shape[1], device=rt.device)[None, :, None, None]
        order = order.expand(batch, -1, rt.shape[2], 1).float() / rt.shape[1]
        route = self.rt_enc(torch.cat((rt[..., :2], order), -1), on)
        route_ok = on.any(-1) & (b["route_ok"] > 0.5)[:, None]
        tok = torch.cat((agents, maps, route), 1)
        ok = torch.cat((agent_ok, map_ok, route_ok), 1)
        pos = _token_positions(b, nbr, mp, kind)
        pe = sine_position(pos, tok.shape[-1])
        idx, near = knn(pos, ~ok, self.cfg.local_neighbours)
        plan = gather_plan(idx, tok.shape[1])
        mem = tok
        for layer in self.enc:
            mem = layer(mem, pe, idx, near, plan)
        a, p = agents.shape[1], maps.shape[1]
        # the center's start velocity (its current recorded velocity, in its frame): known where
        # its current frame is valid and no input dropout cleared it (`v0_ok`, set in training)
        hist = b["hist"].float()
        v0 = hist[:, -1, 6:8]
        v0_ok = b["v0_ok"] > 0.5 if "v0_ok" in b else hist[:, -1, 5] > 0.5
        steps, dense = self.mtr(
            self.ip[b["typ"].long()], mem[:, :a], pos[:, :a], agent_ok, mem[:, a : a + p],
            pos[:, a : a + p], map_ok, mem[:, a + p :], pos[:, a + p :], route_ok, wp.float(),
            v0=v0, v0_ok=v0_ok,
        )  # fmt: skip
        if all_steps:
            return {"steps": steps, "dense": dense}
        return steps[-1]


def _token_positions(b: dict, nbr, mp, kind) -> torch.Tensor:
    """[B, T, 2] position of every encoder input, as MTR: the agent itself at the origin, road
    users at their last position seen, map segments (and route chunks) at their centre."""
    seen = nbr[..., 5] > 0.5  # [B, N, H]
    last = (seen * torch.arange(1, seen.shape[-1] + 1, device=nbr.device)).argmax(-1)
    nb = torch.gather(nbr[..., :2], 2, last[..., None, None].expand(-1, -1, 1, 2))[:, :, 0]
    on = (kind > 0).float()[..., None]
    seg = (mp[..., :2] * on).sum(2) / on.sum(2).clamp(min=1)
    rt = b["route"].float()
    ok = (rt[..., 2] > 0.5).float()[..., None]
    route = (rt[..., :2] * ok).sum(2) / ok.sum(2).clamp(min=1)
    return torch.cat([nb.new_zeros(nb.shape[0], 1, 2), nb, seg, route], 1)


def load_route_model(path, device="cpu") -> RouteModel:
    """A checkpoint written by `train`, with the configuration and feature settings it was
    trained with (`model_config`, `data_preprocess.feature_config`); one of the removed rs_w352
    model or without the route and signal inputs is refused."""
    ck = torch.load(path, map_location="cpu")
    if not (ck.get("use_route") and ck.get("use_sig")):
        raise ValueError(f"{path}: a model without the route and signal inputs is not supported")
    for flag in ("use_gmm", "use_frenet", "use_kin"):  # the prototype's abandoned outputs
        if ck.get(flag):
            raise ValueError(f"{path}: {flag} output is not supported")
    model = RouteModel(model_config(ck["train"]["model"]), ck["points"], history=ck["history"])
    model.load_state_dict(ck["state"])
    model.fcfg = feature_config(ck.get("features"))  # the context it was trained with
    return model.to(device).eval()
