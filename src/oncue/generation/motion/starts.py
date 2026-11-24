"""Build the starts a request is tried from (`request_starts`): moving, stand_go and stop_go,
each with its ideal motion, ranked by how common they are in the logs."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from ...core.config import LIMITS_BY_TYPE
from ...core.kinematics import LIMIT_ROUNDING
from ...utils.tracks import frame_of
from ..checks.validity import placement_reject
from ..context import SceneContext
from ..requests.build import Request, Start
from ..targets import TargetSpec
from .generators import constant_arc, drive_along, pull_arc

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Iterator

STAND_PLACE_STEP_M = 0.5  # stand_go tries a place every this far back from the hiding spot
QUEUE_BACK_M = 1.0  # queue sites: at most this far back, still inside the gap's own lane
# commonness keys this close count as a tie; ties keep the order the starts are made in
COMMONNESS_STEP = 0.05
# the cruising start is skipped when a logged starting speed is within this share of it
SAME_SPEED_SHARE = 0.01
END_SLACK_M = 1e-9  # np.arange stops this far past its end so a place at the end is kept


def request_starts(ctx: SceneContext, req: Request, spec: TargetSpec) -> list[Start]:
    """Return the starts tried for `req`, most common first, at most `spec.starts_per_request`.

    Sets `req.realizable` when the request has no geometry; starts left out are counted by
    reason in `req.start_drops`.
    """
    geometry, reason = request_geometry(ctx, req, spec)
    if geometry is None:
        req.realizable = reason
        return []
    req.mean_speed = geometry.speed
    kept = []
    for start, why in candidate_starts(ctx, req, spec, geometry):
        ctx.consider("start")
        if why is None:
            kept.append(start)
        else:
            ctx.drop("start", why)
            req.start_drops[why] += 1
    ranked = sorted(kept, key=lambda start: commonness(start, spec))
    budget = spec.starts_per_request  # None keeps every start
    cut = 0 if budget is None else len(ranked) - budget
    if cut > 0:
        ctx.drop("start", "over_budget", cut)
        req.start_drops["over_budget"] += cut
    return ranked[:budget]


class RequestGeometry(NamedTuple):
    """Where a request's target must be seen."""

    s_reveal: float  # arc of the reveal point: the visible stretch to the contact begins here
    d_vis: float  # distance from the reveal point to the contact waypoint (m)
    speed: float  # mean speed over d_vis in W (m/s)


def request_geometry(
    ctx: SceneContext, req: Request, spec: TargetSpec
) -> tuple[RequestGeometry | None, str | None]:
    """Return (geometry, None), or (None, why the request is not realizable)."""
    s_reveal = req.table.boundary(req.f_reveal)
    if s_reveal is None or s_reveal >= req.s_contact:
        return None, "cannot_hide"
    d_vis = req.s_contact - s_reveal
    speed = d_vis / ((req.f_contact - req.f_reveal) * ctx.dt)
    limit = LIMITS_BY_TYPE[spec.agent_type].speed
    if limit is not None and speed > limit:
        return None, "beyond_physical_speed"
    if certain_speeding(ctx, req, spec, speed):
        return None, "certain_speeding"
    return RequestGeometry(s_reveal, d_vis, speed), None


def certain_speeding(ctx: SceneContext, req: Request, spec: TargetSpec, speed: float) -> bool:
    """Under the transition rule, return whether the mean speed guarantees a speeding flaw.

    Compares against the highest posted limit along the route's lanes, not the median one, so a
    request on a faster lane is not dropped.
    """
    if spec.start_rule != "transition" or spec.speeding_tol_mps is None:
        return False
    highest = req.route.meta.get("limit_max")
    return highest is not None and speed > highest + spec.speeding_tol_mps


def atypical_speed(speed: float, spec: TargetSpec) -> bool:
    """Return whether a mean speed is outside `spec.log.speed_common_mps` (counted, not
    dropped)."""
    low, high = spec.log.speed_common_mps
    return not low <= speed <= high


def candidate_starts(
    ctx: SceneContext, req: Request, spec: TargetSpec, geometry: RequestGeometry
) -> Iterator[tuple[Start, str | None]]:
    """Yield (start, reason left out or None) for every start kind the route allows."""
    if "moving" in spec.start_kinds:
        if spec.start_rule == "transition":
            yield from transition_starts(ctx, req, spec, geometry)
        else:
            yield moving_start(ctx, req, spec, geometry)
    if "stand_go" in spec.start_kinds:
        yield from stand_go_starts(ctx, req, spec, geometry)
    if "stop_go" in spec.start_kinds and req.route.s_hold is not None:
        yield from stop_go_starts(ctx, req, spec, geometry)


def commonness(start: Start, spec: TargetSpec) -> int:
    """Return a commonness key for `start` (smaller: more common): log distances of its speed
    and start acceleration from the log medians."""
    log = spec.log
    if "v0_middle" in start.extra:  # transition rule
        key = abs(np.log(start.speed / start.extra["v0_middle"]))
        key += abs(start.extra["transition_accel"]) / log.start_accel_median_mps2
    elif start.profile == "moving":
        key = abs(np.log(start.speed / log.speed_median_mps))
    else:
        key = abs(np.log(start.extra["cruise_speed"] / log.speed_median_mps))
        key += abs(np.log(start.accel / log.start_accel_median_mps2))
    return round(key / COMMONNESS_STEP)


# ---------------------------------------------------------------- moving
def moving_start(
    ctx: SceneContext, req: Request, spec: TargetSpec, geometry: RequestGeometry
) -> tuple[Start, str | None]:
    """Return a moving start at the request's mean speed, at the reveal point at the reveal
    frame."""
    anchor, dt = ctx.anchor, ctx.dt
    s_anchor = req.s_contact - geometry.speed * (req.f_contact - anchor) * dt
    start = Start(s_anchor, geometry.speed, profile="moving", s_reveal=geometry.s_reveal, accel=0.0)
    first = spec.first_frame(anchor, dt)
    arc = constant_arc(ctx.n, anchor, first, dt, s_anchor, geometry.speed)
    if arc[first] < 0:
        return start, "outside_route"
    start.states, start.valid = drive_along(req.route.line, ctx.n, dt, arc)
    return start, seen_before_reveal(req, arc, checked_from(ctx, spec))


def transition_starts(
    ctx: SceneContext, req: Request, spec: TargetSpec, geometry: RequestGeometry
) -> Iterator[tuple[Start, str | None]]:
    """Yield (start, reason) for each transition-rule start of `req`.

    Each moves at a starting speed v0 before the anchor, then changes speed evenly while hidden
    to reach the mean speed at the reveal point by the reveal frame.
    """
    speeds = starting_speeds(req, spec)
    middle = speeds[len(speeds) // 2]
    anchor, dt = ctx.anchor, ctx.dt
    hidden = (req.f_reveal - anchor) * dt  # from the anchor to the reveal (s)
    mean = geometry.speed
    # a cruising start at the mean speed, unless that speed is above the common ones
    if (
        mean <= spec.log.speed_common_mps[1]
        and not np.isclose(speeds, mean, rtol=SAME_SPEED_SHARE).any()
    ):
        speeds = (mean, *speeds)
    for v0 in speeds:
        change = (mean - v0) / hidden if hidden > 0 else 0.0  # m/s^2 while hidden
        s_anchor = geometry.s_reveal - (v0 + mean) / 2 * hidden
        start = Start(s_anchor, v0, profile="moving", s_reveal=geometry.s_reveal, accel=0.0)
        start.extra = {"v0_middle": middle, "transition_accel": change}
        if hidden <= 0:
            yield start, "no_hidden_time"
            continue
        limits = LIMITS_BY_TYPE[spec.agent_type]
        capability = limits.accel if change > 0 else limits.brake
        if capability is not None and abs(change) > capability * (1 + LIMIT_ROUNDING):
            yield start, "transition_beyond_physical"
            continue
        first = spec.first_frame(anchor, dt)
        t = (np.arange(ctx.n) - anchor) * dt
        arc = np.where(
            t <= hidden,
            s_anchor + v0 * t + np.where(t > 0, 0.5 * change * t**2, 0.0),
            geometry.s_reveal + mean * (t - hidden),
        )
        arc[:first] = np.nan
        if arc[first] < 0:
            yield start, "outside_route"
            continue
        start.states, start.valid = drive_along(req.route.line, ctx.n, dt, arc)
        yield start, seen_before_reveal(req, arc, checked_from(ctx, spec))


def starting_speeds(req: Request, spec: TargetSpec) -> tuple[float, ...]:
    """Return the transition rule's starting speeds: logged ratios of a lane route's limit, or
    logged speeds."""
    log, limit = spec.log, req.route.meta.get("limit")
    if log.start_limit_ratios and limit:
        return tuple(ratio * limit for ratio in log.start_limit_ratios)
    if not log.start_speeds_mps:
        raise ValueError(f"{spec.kind}: the transition rule needs LogMotion.start_speeds_mps")
    return log.start_speeds_mps


# ---------------------------------------------------------------- from standstill
def stand_go_starts(
    ctx: SceneContext, req: Request, spec: TargetSpec, geometry: RequestGeometry
) -> Iterator[tuple[Start, str | None]]:
    """Yield stand_go starts: standing behind the hiding spot, then off at a logged acceleration."""
    route = req.route
    spot = route.meta["s_stand"]  # arc of the hiding spot
    deepest = min(spot, QUEUE_BACK_M if route.kind == "queue" else spec.stand_back_m)
    for back in np.arange(0.0, deepest + END_SLACK_M, STAND_PLACE_STEP_M):
        s = spot - float(back)
        misplaced = None
        if spec.placement == "kerbside":
            misplaced = placement_reject(ctx, route, route.point(s))
        for accel in spec.log.start_accels_mps2:
            start = Start(
                s, back_m=float(back), profile="stand_go", s_reveal=geometry.s_reveal, accel=accel
            )
            yield start, misplaced or go_profile(ctx, req, spec, start)


def stop_go_starts(
    ctx: SceneContext, req: Request, spec: TargetSpec, geometry: RequestGeometry
) -> Iterator[tuple[Start, str | None]]:
    """Yield stop_go starts: waiting at the route's stop line, then off at a logged acceleration."""
    for accel in spec.log.start_accels_mps2:
        start = Start(req.route.s_hold, profile="stop_go", s_reveal=geometry.s_reveal, accel=accel)
        yield start, go_profile(ctx, req, spec, start)


def go_profile(ctx: SceneContext, req: Request, spec: TargetSpec, start: Start) -> str | None:
    """Give a start from standstill its motion, timed by `go_timing`, or return why it has
    none."""
    dt = ctx.dt
    timing = go_timing(
        start.s_reveal - start.s,
        req.s_contact - start.s,
        (req.f_contact - req.f_reveal) * dt,
        start.accel,
    )
    if timing is None:
        return "no_go_profile"
    lead, cruise = timing  # how long before the reveal it leaves, and the cruise speed
    t_go = req.f_reveal * dt - lead
    start.f_go = frame_of(t_go, dt)
    if start.f_go <= ctx.anchor:
        return "go_in_history"
    start.lead_s = lead
    start.extra = {"cruise_speed": cruise, "t_go": t_go}
    times = np.arange(ctx.n) * dt
    arc = pull_arc(times, start.s, t_go, cruise, start.accel, req.route.line.length)
    arc[: spec.first_frame(ctx.anchor, dt)] = np.nan
    start.states, start.valid = drive_along(req.route.line, ctx.n, dt, arc)
    return seen_before_reveal(req, arc, checked_from(ctx, spec))


def go_timing(
    to_reveal: float, to_contact: float, window: float, accel: float
) -> tuple[float, float] | None:
    """Solve a go-from-standstill timing.

    Returns (seconds before the reveal the target leaves, cruise speed) to cover `to_reveal` by
    the reveal and `to_contact` `window` seconds later, or None when no such motion exists.
    """
    if to_reveal < 0 or to_contact <= to_reveal:
        return None
    mean = (to_contact - to_reveal) / window
    if to_reveal >= mean**2 / (2 * accel):  # cruising at the mean speed by the reveal
        return to_reveal / mean + mean / (2 * accel), mean
    # still speeding up at the reveal
    lead = np.sqrt(2 * to_reveal / accel)
    total = lead + window
    room = total**2 - 2 * to_contact / accel
    if room < 0:
        return None  # late even accelerating all along
    # the smaller root: the cruise speed is reached by the contact
    return float(lead), float(accel * (total - np.sqrt(room)))


# ---------------------------------------------------------------- every start
def checked_from(ctx: SceneContext, spec: TargetSpec) -> int:
    """Return the first frame a start's glimpse check reads: 0, or the anchor when the assumed
    history counts as hidden."""
    return 0 if spec.history_visibility else ctx.anchor


def seen_before_reveal(req: Request, arc: np.ndarray, from_frame: int) -> str | None:
    """Return "seen_before_reveal" when the nearest still copy is visible at a frame from
    `from_frame` up to the reveal, else None."""
    table = req.table
    frames = np.arange(len(arc)) if table.frames is None else np.asarray(table.frames)
    frames = frames[(frames >= from_frame) & (frames < req.f_reveal) & np.isfinite(arc[frames])]
    if not len(frames):
        return None
    copies = np.abs(table.ds[None, :] - arc[frames, None]).argmin(axis=1)
    return "seen_before_reveal" if table.seen[copies, frames].any() else None


# ---------------------------------------------------------------- the grid generator
GRID_GO_STEP_S = 0.5  # start-from-standstill grid: a go time every this long after the anchor
GRID_PLACE_STEP_M = 5.0  # moving grid: a placement every this far before the contact
GRID_HORIZON_S = 8.0  # grid reaches as far back as the median speed covers in this long


def grid_starts(ctx: SceneContext, req: Request, spec: TargetSpec) -> list[Start]:
    """Return `spec.starts_per_request` grid starts spread evenly, at the logged median speed
    and not timed by the geometry."""
    geometry, _ = request_geometry(ctx, req, spec)
    if geometry is None:
        return []
    grid = list(_grid(ctx, req, spec, geometry.s_reveal))
    if spec.starts_per_request is None or len(grid) <= spec.starts_per_request:
        return grid
    picks = np.unique(np.linspace(0, len(grid) - 1, spec.starts_per_request).round().astype(int))
    return [grid[k] for k in picks]


def _grid(ctx: SceneContext, req: Request, spec: TargetSpec, s_reveal: float):
    """Yield every grid start in order (see `grid_starts`)."""
    n, dt, anchor, route, log = ctx.n, ctx.dt, ctx.anchor, req.route, spec.log
    first = spec.first_frame(anchor, dt)
    v, accel = log.speed_median_mps, log.start_accel_median_mps2
    still = route.s_hold if route.s_hold is not None else route.meta.get("s_stand")
    if still is not None:
        kind = "stop_go" if route.s_hold is not None else "stand_go"
        step = frame_of(GRID_GO_STEP_S, dt)
        for f_go in range(anchor + step, req.f_reveal, step):
            arc = pull_arc(np.arange(n) * dt, still, f_go * dt, v, accel, route.line.length)
            arc[:first] = np.nan
            start = Start(still, f_go=f_go, profile=kind, s_reveal=s_reveal, accel=accel)
            start.extra = {"grid": True, "cruise_speed": v}
            start.states, start.valid = drive_along(route.line, n, dt, arc)
            yield start
    for d in np.arange(GRID_PLACE_STEP_M, v * GRID_HORIZON_S + END_SLACK_M, GRID_PLACE_STEP_M):
        s = req.s_contact - d
        arc = constant_arc(n, anchor, first, dt, s, v)
        if arc[first] < 0:
            break
        start = Start(float(s), float(v), profile="moving", s_reveal=s_reveal, accel=0.0)
        start.extra = {"grid": True}
        start.states, start.valid = drive_along(route.line, n, dt, arc)
        yield start
