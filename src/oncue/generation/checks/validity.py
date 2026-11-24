"""Hard checks for a candidate motion: `check_candidate` runs them cheapest first and records
the first that rejects. A candidate is never edited."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import shapely

from ...core.config import LIMITS_BY_TYPE, SiteConfig, VehicleConfig
from ...core.cutoff import Cutoff, cutoff_frame
from ...core.kinematics import beyond_physical, measures
from ...core.types import Scene, Track
from ...core.visibility import RevealEvent, find_reveals, occluders, visibility_values
from ...utils.geometry import (
    box_corners,
    heading_unit_vector,
    step_start_frames,
    swept_overlap,
    wrap_angle,
)
from ...utils.tracks import frame_of
from .realism import flaw_count, sample_labels
from .scene import scene_rules

if TYPE_CHECKING:  # import only for type hints
    from numpy.typing import ArrayLike

    from ...core.config import VisibilityConfig
    from ..context import SceneContext
    from ..requests.build import Request, VisibilityTable
    from ..requests.sites import Route
    from ..targets import TargetSpec

KERBSIDE_SITE_KINDS = ("kerb", "median")  # sites whose standing place is near a road edge
# step for a one-copy visibility table; any positive value maps every arc to the copy
ONE_COPY_STEP_M = 0.25
UNMEASURED = "unmeasured"  # a window too short to measure is its own reject, never a pass


def check_candidate(
    ctx: SceneContext,
    req: Request,
    spec: TargetSpec,
    states: np.ndarray,
    valid: np.ndarray,
    table: VisibilityTable | None = None,
) -> Checked:
    """Run every hard check on one candidate for `req`; a passing one also gets its labels."""
    tol = tolerance_frames(spec, ctx.dt)
    why = _early_reject(ctx, req, spec, states, valid, table, tol)
    if why is not None:
        return Checked(why)
    f_c = contact_frame(ctx, spec, states, valid)
    if f_c is None:
        return Checked("no_contact")
    with_target = inserted(ctx, spec, states, valid)
    motion = motion_checks(ctx, spec, with_target, f_c)
    if motion.reject is not None:
        detail = {"found": motion.found}
        return Checked(motion.reject, f_contact=f_c, info=detail, cutoff=motion.cutoff)
    found = reveal_check(ctx, spec, with_target, f_c)
    if found.reject is not None:
        return found
    event = found.event
    if out_of_range(ctx, with_target, spec, event.reveal):
        return Checked("reveal_out_of_range", event, f_c)
    # diagnostic only: did reveal and contact land at the requested frames?
    on_time = abs(f_c - req.f_contact) <= tol and abs(event.reveal - req.f_reveal) <= tol
    # W itself must be within the tolerance; reveal and contact may shift together
    window = abs(window_error(req, f_c, event.reveal))
    info = {"on_time": on_time, "window_hit": bool(window <= tol)}
    if window > tol:
        return Checked("window", event, f_c, info=info)
    window_scene = inserted(ctx, spec, states, motion.within)
    labels = sample_labels(
        ctx, req.route, spec, window_scene, motion.peaks, motion.moving, req.mean_speed
    )
    info.update(labels)
    return Checked(None, event, f_c, info, motion.cutoff)


@dataclass
class Checked:
    reject: str | None  # name of the first failed hard check; None when all pass
    event: RevealEvent | None = None
    f_contact: int | None = None
    info: dict | None = None
    cutoff: Cutoff | None = None  # the test's cutoff frame and reason (`test_cutoff`)

    @property
    def flaw_count(self) -> int:
        """Return how many flaws were found (0 when the candidate has no labels yet)."""
        return flaw_count((self.info or {}).get("flaws") or {})


def tolerance_frames(spec: TargetSpec, dt: float) -> int:
    """Return the spec's timing tolerance converted to frames."""
    return frame_of(spec.timing_tolerance_s, dt)


def prescreen_passes(err: int | None, tol: int) -> bool:
    """Return whether an approximate W error (`approx_timing_error`) passes the prescreen: at
    most one frame more than the tolerance `tol`."""
    return err is not None and err <= tol + 1


def _early_reject(
    ctx: SceneContext,
    req: Request,
    spec: TargetSpec,
    states: np.ndarray,
    valid: np.ndarray,
    table: VisibilityTable | None,
    tol: int,
) -> str | None:
    """Run the checks that need no inserted scene and return the first failure, or None."""
    anchor, dt = ctx.anchor, ctx.dt
    if not is_finite(states, valid):
        return "nonfinite"
    # the first steps always lie inside the checked window
    end = valid_end(valid, anchor)
    if not start_jump_ok(states, anchor, end, dt, spec):
        return "start_jump"
    if spec.agent_type == "vehicle" and not start_turn_ok(states, anchor, end, VehicleConfig()):
        return "start_turn"
    if spec.prescreen and table is not None:
        err = approx_timing_error(ctx, req, spec, states, valid, table)
        if not prescreen_passes(err, tol):
            return "prescreen"
    return None


def is_finite(states: np.ndarray, valid: np.ndarray) -> bool:
    """Return whether every valid state is finite."""
    return bool(np.isfinite(states[np.asarray(valid, bool)]).all())


def placement_reject(ctx: SceneContext, route: Route, start_xy: np.ndarray) -> str | None:
    """Return a reason a kerbside standing place is implausible, or None."""
    p = shapely.Point(*start_xy)
    if route.kind != "queue":  # a queue gap: the target stands in the gap, on the road
        if ctx.road is not None and ctx.road.contains(p):
            return "on_road"
        # only kerbside sites stand near the edge; a lot's rows lie farther off the road
        near_edge = route.kind in KERBSIDE_SITE_KINDS
        far = ctx.edge_lines is not None and ctx.edge_lines.distance(p) > SiteConfig().stand_edge_m
        if near_edge and far:
            return "far_from_edge"
    if ctx.static is not None and ctx.static.contains(p):
        return "in_static"
    return None


def start_jump_ok(states: np.ndarray, anchor: int, end: int, dt: float, spec: TargetSpec) -> bool:
    """Return whether a vehicle's first-step velocity stays within `spec.start_jump_mps`."""
    if spec.agent_type != "vehicle" or spec.start_jump_mps is None or end <= anchor + 1:
        return True
    step = (states[anchor + 1, :2] - states[anchor, :2]) / dt  # first-step velocity
    h0, v0 = states[anchor, 2], states[anchor, 3]
    return bool(np.linalg.norm(step - v0 * heading_unit_vector(h0)) <= spec.start_jump_mps)


def valid_end(valid: np.ndarray, start: int) -> int:
    """Return the exclusive end of the target's valid run starting at frame `start`."""
    return start + int(np.argmin(np.r_[valid[start:], False]))


def approx_timing_error(
    ctx: SceneContext,
    req: Request,
    spec: TargetSpec,
    states: np.ndarray,
    valid: np.ndarray,
    table: VisibilityTable,
) -> int | None:
    """Return the frames by which the candidate's W misses the requested W, using the table's
    approximate reveal; None without contact or reveal."""
    f_c = contact_frame(ctx, spec, states, valid)
    if f_c is None:
        return None
    fa = approx_reveal(table, req.route, states, f_c, ctx.anchor)
    return None if fa is None else abs(window_error(req, f_c, fa))


def contact_frame(
    ctx: SceneContext, spec: TargetSpec, states: np.ndarray, valid: np.ndarray
) -> int | None:
    """Return the first frame after the anchor the reference ego's box meets the target, or
    None."""
    ego, sim = ctx.reference.ego, ctx.sim
    both = sim[(sim > ctx.anchor) & valid[sim]]
    if not len(both):
        return None
    prev = step_start_frames(both, ctx.anchor, valid, ego.valid)
    sub = ctx.config.oracle.collision_substeps
    hit = swept_overlap(
        ego.states[prev], ego.states[both], ego.size, states[prev], states[both], spec.size, sub
    )
    return int(both[np.argmax(hit)]) if hit.any() else None


def approx_reveal(
    table: VisibilityTable, route: Route, states: np.ndarray, f_contact: int, anchor: int
) -> int | None:
    """Return the last onset of visibility before the contact, from the visibility table; None
    when never seen or glimpsed before it."""
    arcs = shapely.line_locate_point(route.line, shapely.points(states[:, :2]))
    step = table.ds[1] - table.ds[0] if len(table.ds) > 1 else ONE_COPY_STEP_M
    idx = np.clip(np.round((arcs - table.ds[0]) / step).astype(int), 0, len(table.ds) - 1)
    seen = table.seen[idx, np.arange(len(states))]
    if not seen[:f_contact].any():
        return None
    onsets = [f for f in range(anchor + 1, f_contact) if seen[f] and not seen[f - 1]]
    if not onsets or seen[: onsets[-1]].any():
        return None
    return onsets[-1]


def inserted(ctx: SceneContext, spec: TargetSpec, states: np.ndarray, valid: np.ndarray) -> Scene:
    """Return the reference scene with the candidate inserted as the target."""
    target = Track(spec.track_id, spec.agent_type, spec.size, states, valid, spec.height)
    return replace(ctx.reference, tracks={**ctx.reference.tracks, spec.track_id: target})


# ---------------------------------------------------------------- the test's window
@dataclass(frozen=True)
class MotionChecks:
    """Result of the target's motion checks over the test's window."""

    cutoff: Cutoff  # the test's cutoff; the window ends before this frame
    within: np.ndarray  # the target's validity restricted to the window
    peaks: dict | None  # its kinematic peaks within the window (`window_measures`)
    # the first failed window check
    reject: str | None = None
    found: object = None  # what that check found: limits exceeded, a frame, or [id, frame]
    # [[id, frame]] of each moving road user it touches in the window, sorted by frame
    moving: list = field(default_factory=list)


def motion_checks(
    ctx: SceneContext, spec: TargetSpec, with_target: Scene, f_contact: int
) -> MotionChecks:
    """Run the window-level checks on the inserted target, cheapest first, up to a failure."""
    target = with_target.tracks[spec.track_id]
    states, valid = target.states, target.valid
    cutoff = test_cutoff(ctx, spec, states, valid, f_contact)
    within = valid & (np.arange(len(valid)) < cutoff.frame)
    peaks = window_measures(states, valid, ctx.dt, spec, cutoff.frame)
    checked = partial(MotionChecks, cutoff, within, peaks)
    if peaks is None:
        return checked(UNMEASURED)
    beyond = beyond_physical(peaks, LIMITS_BY_TYPE[spec.agent_type])
    if beyond:
        return checked("physical", beyond)
    static = static_frame(ctx.static, states, within, spec.size)
    if static is not None:
        return checked("static_object", static)
    oracle, replay = ctx.config.oracle, ctx.config.replay
    contacts = road_user_contacts(
        with_target, spec.track_id, oracle.collision_substeps, replay.still_mps
    )
    standing = first_contact(contacts, True, cutoff.frame)
    if standing is not None:
        return checked("background_standing", standing)
    return checked(moving=[[i, f] for i, f, still in contacts if not still and f < cutoff.frame])


def test_cutoff(
    ctx: SceneContext, spec: TargetSpec, states: np.ndarray, valid: np.ndarray, f_contact: int
) -> Cutoff:
    """Return the first frame from `f_contact` on where the target is unobserved or leaves the
    ego's route band, else the run's end."""
    band = scene_rules(ctx).band
    return cutoff_frame(band, states, valid, spec.size, f_contact, ctx.n)


def window_measures(
    states: np.ndarray, valid: np.ndarray, dt: float, spec: TargetSpec, cutoff: int
) -> dict | None:
    """Return the target's kinematic peaks before `cutoff`, or None when too short to measure."""
    first = int(np.argmax(valid))
    end = valid_end(valid, first)
    xy = states[first:end, :2]
    return measures(xy, dt, spec.kinematics.moving_mps, upto=cutoff - first)


def static_frame(
    static: shapely.Geometry | None, states: np.ndarray, within: np.ndarray, size: ArrayLike
) -> int | None:
    """Return the first frame of `within` the target's box meets a static object, or None."""
    if static is None:
        return None
    frames = np.flatnonzero(within)
    boxes = shapely.polygons(box_corners(np.asarray(states, float)[frames, :3], size))
    inside = shapely.intersects(boxes, static)
    return int(frames[np.argmax(inside)]) if inside.any() else None


def road_user_contacts(scene: Scene, target_id: str, substeps: int, still_mps: float) -> list:
    """Return [(id, frame, standing)] of road users the target touches from its first valid frame.

    Each road user gives its first contact while moving and its first while standing (slower
    than `still_mps`).
    """
    start = int(np.argmax(scene.tracks[target_id].valid))
    found = []
    for other, hits in _contact_frames(scene, target_id, substeps, start):
        standing = other.states[hits, 3] < still_mps
        for still in (False, True):
            frames = hits[standing == still]
            if len(frames):
                found.append((other.track_id, int(frames[0]), still))
    return sorted(found, key=lambda contact: contact[1])


def _contact_frames(scene: Scene, target_id: str, substeps: int, start: int):
    """Yield (road user, frames it touches the target) for each road user but the ego."""
    track = scene.tracks[target_id]
    frames = np.arange(start, len(scene.timestamps))
    for other in scene.others(target_id):
        ok = frames[track.valid[frames] & other.valid[frames]]
        if not len(ok):
            continue
        prev = step_start_frames(ok, start, track.valid, other.valid)
        swept = swept_overlap(
            track.states[prev],
            track.states[ok],
            track.size,
            other.states[prev],
            other.states[ok],
            other.size,
            substeps,
        )
        hits = ok[swept]
        if len(hits):
            yield other, hits


def first_contact(contacts: list, standing: bool, until: int) -> list | None:
    """Return [id, frame] of the first contact matching `standing` before `until`, or None."""
    return next(([i, f] for i, f, still in contacts if still == standing and f < until), None)


# ---------------------------------------------------------------- the reveal
def reveal_check(
    ctx: SceneContext, spec: TargetSpec, with_target: Scene, f_contact: int
) -> Checked:
    """Find the reveal the window W counts from, seen from the reference ego."""
    vis = ctx.config.visibility
    blockers = occluders(with_target, spec.track_id, vis)
    values = visibility_values(
        with_target.ego, with_target.tracks[spec.track_id], blockers, vis, ctx.static
    )
    if not spec.history_visibility:  # sensitivity run that treats the history as hidden
        before = np.arange(len(values)) < ctx.anchor
        values = np.where(before & np.isfinite(values), 0.0, values)
    why, event = reveal_before_contact(values, vis, ctx.dt, ctx.anchor, f_contact)
    return Checked(why, event, f_contact)


def reveal_before_contact(
    values: np.ndarray, vis: VisibilityConfig, dt: float, anchor: int, f_contact: int
) -> tuple[str | None, RevealEvent | None]:
    """Return (reason, event) for the last reveal after the anchor and before the contact.

    `reason` is None, "no_reveal" (event None), "glimpsed_before" or "reoccluded".
    """
    events = [e for e in find_reveals(values, vis, dt) if anchor < e.reveal < f_contact]
    if not events:
        return "no_reveal", None
    event = events[-1]
    seen = np.isfinite(values) & (values >= vis.threshold)
    if seen[: event.reveal].any():
        return "glimpsed_before", event
    gap = run = 0
    for v in seen[event.reveal : f_contact]:
        run = 0 if v else run + 1
        gap = max(gap, run)
    if gap * dt > SiteConfig().max_gap_s:
        return "reoccluded", event
    return None, event


def out_of_range(ctx: SceneContext, with_target: Scene, spec: TargetSpec, reveal: int) -> bool:
    """Return whether the target is revealed farther than the perception range (centre to
    centre)."""
    gap = with_target.tracks[spec.track_id].states[reveal, :2] - with_target.ego.states[reveal, :2]
    return bool(np.hypot(*gap) > ctx.config.visibility.perception_range_m)


def window_error(req: Request, f_contact: int, f_reveal: int) -> int:
    """Return the frames by which the measured window exceeds the requested W (can be negative)."""
    return (f_contact - f_reveal) - (req.f_contact - req.f_reveal)


# ---------------------------------------------------------------- a vehicle's start
START_TURN_M = (1.0, 3.0)  # chord lengths over which a vehicle's start heading is checked
START_TURN_TOL_RAD = 0.05  # allowance for position noise over that chord (1-2 cm in WOMD)


def start_turn_ok(states: np.ndarray, anchor: int, end: int, vehicle: VehicleConfig) -> bool:
    """Return whether a vehicle leaves along its anchor heading, as far as the kinematic bicycle
    allows.

    Checks the chord from the anchor, since the heading of centimetre-scale steps is noise.
    """
    xy = states[anchor:end, :2]
    chord = xy - xy[0]
    c = np.linalg.norm(chord, axis=1)
    far = np.flatnonzero(c >= START_TURN_M[1])
    window = slice(0, far[0] + 1 if len(far) else len(c))
    look = c[window] >= START_TURN_M[0]
    if not look.any():
        return True
    c, chord = c[window][look], chord[window][look]
    kappa = np.tan(vehicle.max_steering) / vehicle.wheelbase
    slip = np.arctan(vehicle.center_to_rear_axle * kappa)
    off = np.abs(wrap_angle(np.arctan2(chord[:, 1], chord[:, 0]) - states[anchor, 2]))
    return bool(
        (off <= slip + np.arcsin(np.minimum(kappa * c / 2, 1.0)) + START_TURN_TOL_RAD).all()
    )


# ---------------------------------------------------------------- the closed loop's check
def background_contacts(scene: Scene, target_id: str, substeps: int) -> list[tuple[str, int]]:
    """Return [(id, frame)] of each non-ego road user's first contact with the target, sorted
    by frame."""
    contacts = _contact_frames(scene, target_id, substeps, scene.anchor)
    return sorted(((other.track_id, int(hits[0])) for other, hits in contacts), key=lambda c: c[1])
