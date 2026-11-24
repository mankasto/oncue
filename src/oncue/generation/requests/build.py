"""Build a route's test requests, one per contact point and window W (`build_requests`), and the
route's visibility table."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...core.types import Track
from ...core.visibility import occluders, visibility_values
from ...utils.geometry import line_heading
from ...utils.tracks import frame_of
from ..context import SceneContext
from ..targets import TargetSpec
from .sites import Route

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Iterable

CONTACT_AFTER_ANCHOR_S = 1.0  # a contact waypoint must be at least this long after the anchor
CORNER_KEY = "corner"  # key for the corner contact (`ContactRule.corner_overlap_m`)
ARC_TOL_M = 1e-9  # arcs within this count as the same
TABLE_END_SLACK_M = 1e-6  # np.arange stops this far past the table's end so the last copy is kept


def build_requests(ctx: SceneContext, route: Route, spec: TargetSpec) -> list[Request]:
    """Return one request per contact point and W of `route`, with its realizability set."""
    anchor, dt, n = ctx.anchor, ctx.dt, ctx.n
    cover = ctx.config.oracle.cover_contact_s  # oracle sees the log this long past the contact
    f_contact = frame_of(route.t_ego, dt)  # the reference front reaches the route
    contacts = []
    for key, s_contact in contact_points(ctx, route, spec):
        if s_contact <= 0:
            ctx.drop("contact", "at_start")  # no room on the route before the contact
        elif not anchor + frame_of(CONTACT_AFTER_ANCHOR_S, dt) <= f_contact < n:
            ctx.drop("contact", "entry_out_of_range")
        else:
            contacts.append((key, s_contact))
    if not contacts:
        return []
    s_end = max(s_contact for _, s_contact in contacts)
    table = visibility_table(ctx, route, spec, 0.0, s_end, table_frames(ctx, spec, f_contact))
    out = []
    for key, s_contact in contacts:
        for window_s in spec.windows_s:
            f_reveal = f_contact - frame_of(window_s, dt)
            req = Request(route, window_s, key, s_contact, f_contact, f_reveal, table)
            if f_reveal <= anchor + 1:
                req.realizable = "reveal_in_history"
            elif cover is not None and f_contact + frame_of(cover, dt) > n - 1:
                req.realizable = "contact_near_log_end"
            out.append(req)
    return out


@dataclass
class Request:
    route: Route
    W: float
    contact_key: float | str  # impact fraction of the contact rule, or CORNER_KEY
    s_contact: float  # contact waypoint along the route
    f_contact: int  # the contact waypoint's frame: the reference front's arrival
    f_reveal: int
    table: VisibilityTable = field(repr=False)  # the route's table, shared by its requests
    realizable: str = "ok"
    # d_vis / W from `motion.starts.request_geometry`; None when the geometry fails
    mean_speed: float | None = None
    starts: list[Start] = field(default_factory=list)  # in the order they are tried
    start_drops: Counter = field(default_factory=Counter)  # candidate starts left out, by reason


@dataclass
class VisibilityTable:
    ds: np.ndarray  # arcs along the route
    seen: np.ndarray  # [D, F] whether the still copy at ds[i] is visible at frame f
    frames: np.ndarray | None = None  # frames computed, or None for every frame

    def _column(self, f: int) -> np.ndarray:
        if self.frames is not None and f not in self.frames:
            raise ValueError(f"visibility table: frame {f} was not computed")
        return self.seen[:, f]

    def index(self, s: float) -> int:
        """Return the index of the copy nearest arc `s`."""
        return int(np.argmin(np.abs(self.ds - s)))

    def front(self, f: int, from_s: float | None = None) -> float | None:
        """Return the first arc visible at frame `f`, at or beyond `from_s` (None: the start);
        None when nothing is visible."""
        now = self._column(f)
        if from_s is not None:
            now = now & (self.ds >= from_s - ARC_TOL_M)
        return None if not now.any() else float(self.ds[int(np.argmax(now))])

    def boundary(self, f: int) -> float | None:
        """Return where the last visible stretch begins at frame `f`; None when nothing is
        visible or it begins at the table's start."""
        now = self._column(f)
        if not now.any():
            return None
        first = int(np.flatnonzero(now)[-1])
        while first > 0 and now[first - 1]:
            first -= 1
        return None if first == 0 else float(self.ds[first])


@dataclass
class Start:
    """Where the target is at the anchor and how it leaves: moving at `speed`, or standing at `s`
    until frame `f_go` behind the hiding spot (stand_go) or at the stop line (stop_go)."""

    s: float
    speed: float = 0.0
    f_go: int | None = None
    lead_s: float | None = None  # standing: how long it walks hidden before the reveal
    back_m: float | None = None  # standing: how far behind the hiding spot it stands
    profile: str = "stand_go"  # moving | stand_go | stop_go
    s_reveal: float | None = None  # its reveal waypoint: the first arc visible from s then
    accel: float | None = None  # start acceleration from standstill (0 when moving)
    states: np.ndarray | None = None  # moving: constant-speed profile (the rule profile)
    valid: np.ndarray | None = None
    extra: dict = field(default_factory=dict)  # cruise_speed, t_go, grid, and its kind labels


def contact_points(
    ctx: SceneContext, route: Route, spec: TargetSpec
) -> list[tuple[float | str, float]]:
    """Return [(contact key, arc of the contact waypoint)]: one per impact fraction, then the
    corner contact when the rule has one."""
    rule, ego_w = spec.contact, ctx.reference.ego.size[1]
    # (key, offset inward from the ego's near side, how far ahead of the target's centre)
    wanted = [(frac, frac * ego_w, spec.impact_ahead_m) for frac in rule.impact]
    if rule.corner_overlap_m is not None:  # leading edge, half its length ahead of its centre
        wanted.append((CORNER_KEY, rule.corner_overlap_m, spec.size[0] / 2))
    out = []
    ctx.consider("contact", len(wanted))
    for key, inward, ahead_m in wanted:
        lateral = route.side * (ego_w / 2 - inward)
        s = route.s_at_lateral(ctx.path, lateral, ahead_m=ahead_m)
        if s is not None:
            out.append((key, s))
        else:
            ctx.drop("contact", "not_reached")
    return out


def visibility_table(
    ctx: SceneContext,
    route: Route,
    spec: TargetSpec,
    s0: float,
    s1: float,
    frames: Iterable[int] | None = None,
) -> VisibilityTable:
    """Build the visibility table of `route` from `s0` to `s1`: whether the reference ego sees a
    still copy of the target every `spec.table_step_m`, at `frames` (None: every frame)."""
    vis, n = ctx.config.visibility, ctx.n
    ds = np.arange(s0, s1 + TABLE_END_SLACK_M, spec.table_step_m)
    fr = np.arange(n) if frames is None else np.unique([f for f in frames if 0 <= f < n])
    seen = np.zeros((len(ds), n), bool)
    if not len(ds) or not len(fr):
        return VisibilityTable(ds, seen, None if frames is None else fr)
    poses = []
    for d in ds:
        p = route.point(d)
        if route.kind == "lane":
            h = line_heading(route.line, d, back=0.0, ahead=0.5)
        else:
            h = route.heading(d)
        poses.append([*p, h, 0.0])

    def tiled(track: Track) -> Track:  # the track at frames `fr`, once per copy
        k = len(ds)
        return track.with_states(np.tile(track.states[fr], (k, 1)), np.tile(track.valid[fr], k))

    ref = ctx.reference
    ego = tiled(ref.ego)
    copies = target_track(spec, np.repeat(np.asarray(poses), len(fr), axis=0))
    blockers = [tiled(b) for b in occluders(ref, spec.track_id, vis)]
    v = visibility_values(ego, copies, blockers, vis, ctx.static)
    seen[:, fr] = (np.isfinite(v) & (v >= vis.threshold)).reshape(len(ds), len(fr))
    return VisibilityTable(ds, seen, None if frames is None else fr)


def target_track(spec: TargetSpec, states: np.ndarray, valid: np.ndarray | None = None) -> Track:
    n = len(states)
    return Track(
        spec.track_id,
        spec.agent_type,
        spec.size,
        np.asarray(states, float),
        np.ones(n, bool) if valid is None else np.asarray(valid, bool),
        spec.height,
    )


def table_frames(ctx: SceneContext, spec: TargetSpec, f_contact: int) -> list[int] | None:
    """Return the frames the requests read from the visibility table; None (every frame) with a
    prescreen."""
    if spec.prescreen:
        return None
    return [ctx.anchor, ctx.anchor + 1] + [
        f_contact - frame_of(window_s, ctx.dt) for window_s in spec.windows_s
    ]


def route_limit(ctx: SceneContext, route: Route) -> float:
    """Return the target route's speed limit: its lanes' for a lane route, else the ego's road's."""
    return route.meta.get("limit") or ctx.limit
