"""Candidate motions for a request: the learned generator (`model_candidates`), the rule
generator (`rule_candidates`) and the route profiles the starts are built with."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np
import shapely
from shapely.ops import substring

from ...core.kinematics import polyline_yaw
from ...core.types import STATE_SIZE, Track
from ..context import SceneContext
from ..requests.build import Request, Start, target_track
from ..targets import TargetSpec

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Iterator, Sequence

    from numpy.typing import ArrayLike

    from ...models.modeling.model import RouteModel
    from ..requests.sites import Route

__all__ = [
    "Candidate",
    "anchor_only",
    "constant_arc",
    "drive_along",
    "model_candidates",
    "pull_arc",
    "rolling_target",
    "rule_candidates",
    "start_anchor_state",
    "start_metadata",
]

# drive_along's heading: toward the point this far ahead, or back near the line's end
DRIVE_HEADING_STEP_M = 0.5
MOVING_HEADING_MPS = 0.3  # above this speed, the heading follows each step's direction ("step")
MODEL_HISTORY_FRAMES = 10  # the model reads this many frames before its anchor (1.1 s with it)
WAYPOINT_FEATURES = 6  # a waypoint's model input: x, y, time, two zeros, given


# ---------------------------------------------------------------- learned generator
def model_candidates(
    model: RouteModel,
    ctx: SceneContext,
    req: Request,
    start: Start,
    spec: TargetSpec,
    waypoints: tuple[str, ...] = ("contact", "reveal"),
) -> Iterator[Candidate]:
    """Yield the model's candidates for `req` from `start`, best score first.

    `waypoints` picks which of "contact" and "reveal" the model is given.
    """
    unknown = set(waypoints) - {"contact", "reveal"}
    if unknown:
        raise ValueError(f"waypoints {sorted(unknown)}: each one of contact, reveal")
    import torch

    from ...models.datasets.dataset import waypoint_time
    from ...models.inputs import to_local, to_world, track_inputs

    route, n, dt, anchor = req.route, ctx.n, ctx.dt, ctx.anchor
    first = spec.first_frame(anchor, dt)
    state = start_anchor_state(start, anchor)
    if start.profile == "moving":  # read at the anchor, assumed history before it
        scene = ctx.reference
        track = rolling_target(ctx, route, state, start.s, spec)
        track = track if spec.model_history else anchor_only(track, anchor, spec)
        before = start.states[:anchor, :2]
    elif start.f_go is None:
        return
    else:  # from standstill: read where it leaves, standing there since its first frame
        scene = replace(ctx.reference, anchor=int(start.f_go))
        track = target_track(spec, np.tile(state, (n, 1)), np.arange(n) >= first)
        before = track.states[: scene.anchor, :2]
    model_anchor = scene.anchor
    if model_anchor < MODEL_HISTORY_FRAMES:  # too little history before it
        return
    b, origin, h0 = track_inputs(scene, track, model, model_route(route, start))
    # [contact, reveal, unused]; a waypoint not given stays zero
    wp = torch.zeros(1, 3, WAYPOINT_FEATURES)
    if "contact" in waypoints:
        pc = to_local(route.point(req.s_contact), origin, h0)
        wp[0, 0] = torch.tensor([*pc, waypoint_time(req.f_contact - model_anchor), 0, 0, 1])
    if "reveal" in waypoints:
        pr = to_local(route.point(start.s_reveal), origin, h0)
        wp[0, 1] = torch.tensor([*pr, waypoint_time(req.f_reveal - model_anchor), 0, 0, 1])
    device = next(model.parameters()).device  # inputs are built on the CPU
    with torch.no_grad():
        traj, score = model({key: value.to(device) for key, value in b.items()}, wp.to(device))
    traj, score = traj[0].cpu().numpy(), score[0].cpu().numpy()
    valid = np.arange(n) >= first
    for k in np.argsort(-score):
        glob = to_world(traj[k][: n - model_anchor - 1], origin, h0)
        xy = np.concatenate([before, origin[None], glob], 0)[:n]
        if len(xy) < n:  # scene longer than the model's horizon: hold the last position
            xy = np.concatenate([xy, np.tile(xy[-1], (n - len(xy), 1))], 0)
        states = states_from_positions(xy, model_anchor, dt, spec, state[2], state[3])
        if start.profile == "moving":
            states[first:anchor] = start.states[first:anchor]  # rule's assumed history
        yield Candidate(states, valid, int(k), float(score[k]), start_metadata(route, start))


def start_anchor_state(start: Start, anchor: int) -> list[float]:
    """Return the target's [x, y, heading, speed] at the anchor from `start`."""
    return [*start.states[anchor, :3], start.speed]


def start_metadata(route: Route, start: Start) -> dict:
    """Return the design fields a candidate records about its test."""
    out = {"start_kind": start.profile, "route_kind": route.route_kind, "site_kind": route.kind}
    if "transition_accel" in start.extra:
        out["start_v0"] = round(float(start.speed), 3)
        out["transition_accel"] = round(float(start.extra["transition_accel"]), 3)
    return out


def model_route(route: Route, start: Start) -> np.ndarray:
    """Return the route polyline given to the model: the lane chain, or the rest of the route
    from the start."""
    if "points" in route.meta:
        return np.asarray(route.meta["points"])
    return shapely.get_coordinates(substring(route.line, start.s, route.line.length))


def rolling_target(
    ctx: SceneContext, route: Route, anchor_state: Sequence[float], start_s: float, spec: TargetSpec
) -> Track:
    """Return a target at `anchor_state` with its assumed history before it, unobserved
    after."""
    n, anchor, dt = ctx.n, ctx.anchor, ctx.dt
    arc = constant_arc(n, anchor, spec.first_frame(anchor, dt), dt, start_s, anchor_state[3])
    arc[anchor:] = np.nan
    states, valid = drive_along(route.line, n, dt, arc)
    states[anchor], valid[anchor] = anchor_state, True
    return target_track(spec, states, valid)


def anchor_only(track: Track, anchor: int, spec: TargetSpec) -> Track:
    """Return `track` observed only at frame `anchor`."""
    valid = np.zeros(len(track.valid), bool)
    valid[anchor] = True
    return target_track(spec, np.where(valid[:, None], track.states, 0.0), valid)


# ---------------------------------------------------------------- states from positions
def states_from_positions(
    xy: ArrayLike, anchor: int, dt: float, spec: TargetSpec, heading0: float, speed0: float
) -> np.ndarray:
    """Turn positions [F, 2] into states [F, 4] (x, y, heading, speed) by
    `spec.heading_method`."""
    xy = np.asarray(xy, float)
    n = len(xy)
    st = np.zeros((n, STATE_SIZE))
    st[:, :2] = xy
    if spec.heading_method == "step":
        step = np.diff(xy, axis=0, prepend=xy[:1])
        speed = np.linalg.norm(step, axis=1) / dt
        moving = speed > MOVING_HEADING_MPS
        st[:, 2] = np.where(moving, np.arctan2(step[:, 1], step[:, 0]), heading0)
        st[:, 3] = speed
        return st
    fut = xy[anchor:]
    st[anchor:, 2] = polyline_yaw(fut, still=heading0)
    st[anchor:, 3] = np.linalg.norm(np.diff(fut, axis=0, prepend=fut[:1]), axis=1) / dt
    st[anchor, 2:] = heading0, speed0
    st[:anchor] = st[anchor]
    return st


@dataclass
class Candidate:
    states: np.ndarray  # [F, 4] x, y, heading, speed
    valid: np.ndarray  # [F]
    mode: int | None = None
    score: float | None = None
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------- rule-based generator (baseline)
def rule_candidates(req: Request, start: Start) -> Iterator[Candidate]:
    """Yield the rule-based candidate for `start`: its ideal motion with history."""
    yield Candidate(start.states, start.valid, extra=start_metadata(req.route, start))


# ---------------------------------------------------------------- route profiles
def constant_arc(
    n: int, anchor: int, first: int, dt: float, s_anchor: float, speed: float
) -> np.ndarray:
    """Return [n] arcs at constant `speed` through `s_anchor` at the anchor; NaN before
    `first`."""
    arc = np.full(n, np.nan)
    arc[first:] = s_anchor + speed * (np.arange(first, n) - anchor) * dt
    return arc


def drive_along(
    line: shapely.LineString, n: int, dt: float, s_of_t: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return ([n, 4] states, valid) along `line` at arc `s_of_t[i]` each frame; a frame off
    the line is invalid."""
    states, valid = np.zeros((n, STATE_SIZE)), np.zeros(n, bool)
    for i in range(n):
        s = s_of_t[i]
        if not np.isfinite(s) or s < 0 or s > line.length:
            continue
        p = np.array(line.interpolate(s).coords[0])
        step = DRIVE_HEADING_STEP_M
        if s + step <= line.length:  # heading from a step ahead; near the end, from a step behind
            a, b = p, np.array(line.interpolate(s + step).coords[0])
        else:
            a, b = np.array(line.interpolate(max(s - step, 0.0)).coords[0]), p
        states[i] = [*p, np.arctan2(*(b - a)[::-1]), 0.0]
        valid[i] = True
    states[valid, 3] = np.abs(np.gradient(s_of_t[valid], dt)) if valid.sum() > 1 else 0
    return states, valid


def pull_arc(
    t: np.ndarray, s_hold: float, t_start: float, vmax: float, acc: float, length: float
) -> np.ndarray:
    """Return arc positions: standing at s_hold until t_start, then accelerating to vmax."""
    tau = np.maximum(t - t_start, 0.0)
    ramp = vmax / acc
    s = np.where(tau <= ramp, 0.5 * acc * tau**2, 0.5 * acc * ramp**2 + vmax * (tau - ramp))
    return np.minimum(s_hold + s, length)
