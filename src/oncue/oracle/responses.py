"""The oracle's candidate list and the desired motions it produces.

``candidate_list`` builds the list; ``build`` lays each candidate out along the ego's logged
path, station from its longitudinal plan and offset from its lateral maneuver.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import shapely
from scipy.interpolate import BPoly

from ..core.config import VEHICLE_LIMITS, CandidateConfig, Lateral, Longitudinal, OracleConfig
from ..core.types import STATE_SIZE, Scene, Track
from ..data.map_geometry import lane_continuation, lane_width
from ..utils.geometry import (
    MIN_LINE_POINTS,
    Polyline,
    box_corners,
    cumulative_arc_length,
    distinct_points_mask,
    heading_unit_vector,
    thin_path,
    union_all,
    wrap_angle,
)
from ..utils.tracks import TIME_TOL_S

if TYPE_CHECKING:  # for annotations only: no new import when running
    from collections.abc import Iterable, Sequence

    from numpy.typing import ArrayLike

MAX_INTEGRATION_STEP_S = 0.005  # longitudinal_profile's max integration step
FAMILIES = ("keep", "nudge", "swerve", "lane_change")  # lateral families the refinement turns among
CONFLICT_STEP_M = 0.5
CONFLICT_SEARCH_ACCEL_MPS2 = 3.0  # conflict search reaches as far as this accel allows
STOP_TIME_MARGIN_S = 3.0  # extra simulation time past the weakest braking's stop
STOP_ACCEL_BISECTIONS = 20  # bisection steps when searching for the stopping deceleration
STOP_SEARCH_STEP_S = 0.05  # sampling cadence of the simulated stop
SUBSTEPS_PER_INTERVAL = 4  # minimum integration steps between two user-requested times
CLOCK_TOL_S = 1e-12  # ``longitudinal_profile``'s clock treats a time within this as "here"
PROFILE_CACHE_SIZE = 5000  # profiles the cache holds before it is cleared
STANDSTILL_MPS = 1e-9  # below this speed the ego is treated as standing
# A desired state faster than this heads along its velocity; slower, it keeps the path's heading.
VELOCITY_HEADING_MPS = 1e-3
DIGEST_HEX_DIGITS = 16  # take this many hex digits from a list's sha256 as its digest
MIN_SPLIT_SPAN = 2  # ``_spread`` only halves a span this wide or wider (a rank must fit between)
# Below this speed the ego is as good as stopped: no stop at the conflict is planned.
STOP_PLAN_MIN_MPS = 0.5
JOIN_M = 15.0  # length of the continuation replaced by a smooth join to the logged path
JOIN_STEP_M = 0.5  # spacing of the join's sampled points
END_CURVATURE_M = 2.0  # curvature at a path end is read over this length of the path


# ---------------------------------------------------------------- the candidate list
def candidate_list(cfg: CandidateConfig) -> tuple[Candidate, ...]:
    """Return the oracle's candidates in their fixed order: the core, then the refinement.

    The refinement's mirror packs take turns by lateral family, coarse to fine within each.
    """
    core = [(lon, lat, 0.0) for lon in cfg.longitudinal for lat in cfg.lateral]
    core += [
        (lon, lat, cfg.delay_s)
        for lon in cfg.longitudinal
        if not lon.stop_at_conflict
        for lat in cfg.lateral
        if lat.offset_m != 0
    ]
    core += [(lon, lat, 0.0) for lon in cfg.recover for lat in cfg.lateral]
    core += [(lon, lat, 0.0) for lon in cfg.steer_longitudinal for lat in cfg.steer]
    ids = [candidate_id(*c) for c in core]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate list: the core repeats an entry")
    seen, extra = set(ids), []
    refine = [(lon, lat, 0.0) for lon in cfg.refine_longitudinal for lat in cfg.refine_lateral]
    refine += [
        (lon, lat, cfg.delay_s)
        for lon in cfg.refine_delayed_longitudinal
        for lat in cfg.refine_delayed_lateral
    ]
    for c in refine:
        k = candidate_id(*c)
        if k not in seen:
            seen.add(k)
            extra.append(c)
    out = [
        Candidate(*c, id=k, rank=i, pack=i, core=True)
        for i, (c, k) in enumerate(zip(core, ids, strict=True))
    ]
    families = _mirror_packs(extra)
    packs = _take_turns(_coarse_to_fine(families[f], cfg.refine_longitudinal) for f in FAMILIES)
    for pack in packs:
        first = len(out)
        for c in pack:
            out.append(Candidate(*c, id=candidate_id(*c), rank=len(out), pack=first, core=False))
    return tuple(out)


@dataclass(frozen=True)
class Candidate:
    """One entry of the candidate list.

    Attributes:
        delay_s: how many seconds after the plan the lateral starts.
        id: stable id naming every parameter.
        rank: position in the list.
        pack: rank of the first entry of this candidate's pack (it and its mirror image).
        core: whether the entry belongs to the core.
    """

    lon: Longitudinal
    lat: Lateral
    delay_s: float
    id: str
    rank: int
    pack: int
    core: bool


def candidate_id(lon: Longitudinal, lat: Lateral, delay_s: float) -> str:
    """Return a stable string id naming every parameter of a candidate."""
    return f"{_lon_id(lon)}|{_lat_id(lat)}|+{delay_s:g}s"


def _lon_id(lon: Longitudinal) -> str:
    """Return a compact string id for a longitudinal plan."""
    if lon.stop_at_conflict:
        lo, hi = lon.accel_range
        return f"{lon.name}[{lo:g}..{hi:g},margin{lon.stop_margin_m:g},j{lon.jerk:g}]"
    out = f"{lon.name}[{lon.accel:+g},j{lon.jerk:g}"
    if lon.switch_s is not None:
        out += f",{lon.switch_s:g}s"
    if lon.wait_s is not None:
        out += f",stop+{lon.wait_s:g}s"
    if lon.then_accel is not None:
        out += f">{lon.then_accel:+g},j{lon.then_jerk:g}"
    return out + "]"


def _lat_id(lat: Lateral) -> str:
    """Return a compact string id for a lateral maneuver."""
    if lat.mode == "steer":
        back = "held" if lat.return_s is None else f"back{lat.return_s:g}s"
        return f"steer[{lat.steer_rad:+g}rad,{back}]"
    if lat.offset_m == 0:
        return "keep"
    unit = "lane" if lat.in_lanes else "m"
    return f"{lat.name}[{lat.offset_m:+g}{unit},{lat.duration_s:g}s,{lat.mode}]"


def _mirror_packs(
    entries: list[tuple[Longitudinal, Lateral, float]],
) -> dict[str, list[list[tuple]]]:
    """Group refinement entries into mirror packs (left first) per lateral family."""
    groups: dict = {f: {} for f in FAMILIES}
    for lon, lat, delay in entries:
        family = "keep" if lat.offset_m == 0 else lat.name
        if family not in groups:
            raise ValueError(f"refinement: lateral family {family} not in {FAMILIES}")
        mirror = candidate_id(lon, replace(lat, offset_m=abs(lat.offset_m)), delay)
        groups[family].setdefault(mirror, []).append((lon, lat, delay))
    return {
        f: [sorted(p, key=lambda e: -e[1].offset_m) for p in g.values()] for f, g in groups.items()
    }


def _take_turns(queues: Iterable[list[list[tuple]]]) -> list:
    """Interleave the families' packs, one pack per family per turn."""
    queues = [list(q) for q in queues]
    out = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def _coarse_to_fine(
    packs: list[list[tuple]], longitudinals: tuple[Longitudinal, ...]
) -> list[list[tuple]]:
    """Order one family's packs coarse to fine by longitudinal, offset, duration, then delay."""
    if not packs:
        return []
    lon = _spread(range(len(longitudinals)))
    size = _spread(abs(p[0][1].offset_m) for p in packs)
    span = _spread(p[0][1].duration_s for p in packs)
    return sorted(
        packs,
        key=lambda p: (
            lon[longitudinals.index(p[0][0])],
            size[abs(p[0][1].offset_m)],
            span[p[0][1].duration_s],
            p[0][2],
        ),
    )


def _spread(values: Iterable[float]) -> dict:
    """Return a coarse-to-fine rank per distinct value: the extremes, then repeated midpoints."""
    v = sorted(set(values))
    order = [0] if len(v) == 1 else [0, len(v) - 1]
    todo = [(0, len(v) - 1)]
    while todo:
        lo, hi = todo.pop(0)
        if hi - lo >= MIN_SPLIT_SPAN:
            mid = (lo + hi) // 2
            order.append(mid)
            todo += [(lo, mid), (mid, hi)]
    return {v[i]: k for k, i in enumerate(order)}


def prefix(candidates: Sequence[Candidate], budget: int) -> tuple[Candidate, ...]:
    """Return the first candidates that fit in ``budget``, taking only whole packs."""
    end = 0
    while end < len(candidates):
        stop = end
        while stop < len(candidates) and candidates[stop].pack == candidates[end].pack:
            stop += 1
        if stop > budget:
            break
        end = stop
    return tuple(candidates[:end])


def list_digest(candidates: Iterable[Candidate]) -> str:
    """Return a 16-hex-digit fingerprint of the candidate list."""
    return hashlib.sha256("\n".join(c.id for c in candidates).encode()).hexdigest()[
        :DIGEST_HEX_DIGITS
    ]


# ---------------------------------------------------------------- desired motions
def build(
    scene: Scene,
    start: int,
    end: int,
    candidates: Sequence[Candidate],
    cfg: OracleConfig,
    target: Track,
    profiles: dict | None = None,
) -> Built:
    """Build the desired motions over ``[start, end)`` for every applicable candidate.

    A stop at the conflict does not apply with no conflict ahead or the ego nearly stopped.

    Args:
        profiles: optional longitudinal-profile cache (``cached_profile``); None disables it.
    """
    ego = scene.ego.states
    times = scene.timestamps[start:end] - scene.timestamps[start]
    v0 = float(ego[start, 3])
    accel0 = (
        (ego[start, 3] - ego[start - 1, 3]) / scene.dt
        if start > 0 and scene.ego.valid[start - 1]
        else 0.0
    )
    path = EgoPath(scene, start, cfg.path_extension)
    lc = cfg.candidates
    width = lane_width(scene, ego[start, :2], ego[start, 2], lc.default_lane_width_m)
    plans: dict = {}  # longitudinal id -> (station, speed, path frame), or None (not applicable)
    solved: dict = {}
    rows, desired, commands, accels = [], [], [], []
    for i, c in enumerate(candidates):
        key = _lon_id(c.lon)
        if key not in plans:
            lon = c.lon
            if lon.stop_at_conflict:
                accel = _stop_accel(
                    scene, start, end, path, target, lon, v0, accel0, cfg.max_speed, profiles
                )
                if accel is not None:
                    solved[key] = accel
                lon = None if accel is None else replace(lon, accel=accel)
            if lon is None:
                plans[key] = None
            else:
                station, speed, _ = cached_profile(profiles, v0, accel0, lon, times, cfg.max_speed)
                plans[key] = (station, speed, path.frame(station))
        if plans[key] is None:
            continue
        station, speed, frame = plans[key]
        lat = c.lat
        command = np.full(len(times), np.nan)
        if lat.mode == "steer":
            d = slope = np.zeros_like(station)
            back = np.inf if lat.return_s is None else lat.return_s
            command = np.where(times < back - TIME_TOL_S, lat.steer_rad, 0.0)
        else:
            begin = float(np.interp(c.delay_s, times, station)) if c.delay_s else 0.0
            length = lat.duration_s * max(v0, lc.min_maneuver_speed)
            d, slope = lateral_profile(lat, station - begin, length, width)
        rows.append(i)
        accels.append(solved.get(key))
        desired.append(_compose(frame, station, speed, d, slope, ego[start]))
        commands.append(command)
    shape = (0, len(times))
    return Built(
        tuple(rows),
        np.stack(desired) if desired else np.empty((*shape, STATE_SIZE)),
        np.stack(commands) if commands else np.empty(shape),
        tuple(accels),
    )


class EgoPath:
    """The logged ego path from takeover on, extended ``EXTENSION_M`` past the log.

    The extension goes straight, or with ``extension="lanes"`` along the lane graph where a lane
    continues.
    """

    EXTENSION_M = 200.0

    def __init__(self, scene: Scene, start: int, extension: str = "straight"):
        ego = scene.ego
        stop = ego.valid_until(start, len(scene.timestamps))
        states = ego.states[start:stop]
        heading = np.unwrap(states[:, 2])
        keep = thin_path(states[:, :2])
        points, heading = states[keep, :2], heading[keep]
        tail = None
        if extension == "lanes":
            tail = lane_continuation(scene, points[-1], heading[-1], self.EXTENSION_M)
        if tail is None:
            tail = points[-1] + self.EXTENSION_M * np.array(
                [[np.cos(heading[-1]), np.sin(heading[-1])]]
            )
        tail = _joined(points, heading, tail)
        steps = np.diff(np.vstack((points[-1:], tail)), axis=0)
        tail_heading = np.unwrap(np.r_[heading[-1], np.arctan2(steps[:, 1], steps[:, 0])])[1:]
        allp, allh = np.vstack((points, tail)), np.r_[heading, tail_heading]
        keep = distinct_points_mask(allp)
        self.line = Polyline(allp[keep])
        self.logged_length = float(self.line.stations[int(keep[: len(points)].sum()) - 1])
        self._heading_stations = self.line.stations
        self._headings = allh[keep]
        has_line = len(points) >= MIN_LINE_POINTS  # a curvature needs a line
        curvature = np.gradient(self._headings, self._heading_stations) if has_line else None
        self._curvature = np.zeros_like(self._headings) if curvature is None else curvature

    def frame(self, station: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(position, heading, curvature)`` at each station along the path."""
        heading = np.interp(station, self._heading_stations, self._headings)
        curvature = np.interp(station, self._heading_stations, self._curvature)
        return self.line.position(station), heading, curvature


def _joined(points: np.ndarray, headings: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Smooth the join between the logged path and its continuation ``tail``.

    The first ``JOIN_M`` metres become a quintic matching heading and curvature at both ends.
    """
    ahead = np.vstack((points[-1:], tail))
    if cumulative_arc_length(ahead)[-1] <= JOIN_M + END_CURVATURE_M:
        return tail
    line = Polyline(ahead)
    end = line.position(JOIN_M)
    end_heading, end_curvature = _direction(line, JOIN_M)
    start_curvature = _end_curvature(points, headings)
    span = float(np.linalg.norm(end - points[-1]))
    curve = BPoly.from_derivatives(
        [0.0, span],
        [
            [
                points[-1],
                heading_unit_vector(headings[-1]),
                start_curvature * _normal(headings[-1]),
            ],
            [end, heading_unit_vector(end_heading), end_curvature * _normal(end_heading)],
        ],
    )
    # Skip points within half a step of ``end``: a tiny step would spike the curvature.
    join = curve(np.arange(JOIN_STEP_M, span - JOIN_STEP_M / 2, JOIN_STEP_M))
    rest = line.points[line.stations > JOIN_M + JOIN_STEP_M / 2]
    return np.vstack((join, end[None], rest))


def _direction(line: Polyline, station: float) -> tuple[float, float]:
    """Return ``line``'s heading and curvature at ``station`` from half-END_CURVATURE_M chords."""
    half = END_CURVATURE_M / 2
    back, here, ahead = line.position(np.array([station - half, station, station + half]))
    before = np.arctan2(*(here - back)[::-1])
    turn = float(wrap_angle(np.arctan2(*(ahead - here)[::-1]) - before))
    return before + turn / 2, turn / half


def _end_curvature(points: np.ndarray, headings: np.ndarray) -> float:
    """Return the logged path's curvature over its last ``END_CURVATURE_M`` metres."""
    stations = cumulative_arc_length(points)
    first = int(np.searchsorted(stations, stations[-1] - END_CURVATURE_M, side="right")) - 1
    first = max(first, 0)
    length = stations[-1] - stations[first]
    return 0.0 if length <= 0 else float((headings[-1] - headings[first]) / length)


def _normal(heading: float) -> np.ndarray:
    """Return the left-pointing unit vector perpendicular to ``heading``."""
    return heading_unit_vector(heading + np.pi / 2)


def _stop_accel(
    scene: Scene,
    start: int,
    end: int,
    path: EgoPath,
    target: Track,
    lon: Longitudinal,
    v0: float,
    accel0: float,
    max_speed: float,
    profiles: dict | None,
):
    """Return a deceleration that stops ``lon.stop_margin_m`` short of the conflict, or None.

    It is bisected within ``lon.accel_range``.
    """
    if v0 < STOP_PLAN_MIN_MPS:
        return None
    horizon = (end - 1 - start) * scene.dt
    reach = v0 * horizon + CONFLICT_SEARCH_ACCEL_MPS2 / 2 * horizon**2
    conflict = conflict_station(scene, start, end, path, target, reach)
    if conflict is None:
        return None
    goal = conflict - lon.stop_margin_m
    times = np.arange(0.0, v0 / -lon.accel_range[1] + STOP_TIME_MARGIN_S, STOP_SEARCH_STEP_S)

    def stop_distance(accel: float) -> float:
        station, _, _ = cached_profile(
            profiles, v0, accel0, replace(lon, accel=accel), times, max_speed
        )
        return station[-1]

    low, high = lon.accel_range  # low = strongest braking, high = weakest braking
    if stop_distance(low) >= goal:
        return low
    if stop_distance(high) <= goal:
        return high
    # A mid that overshoots the goal becomes the new strong end, else the new weak end.
    for _ in range(STOP_ACCEL_BISECTIONS):
        mid = (low + high) / 2
        low, high = (low, mid) if stop_distance(mid) > goal else (mid, high)
    return (low + high) / 2


def conflict_station(
    scene: Scene, start: int, end: int, path: EgoPath, target: Track, reach: float
) -> float | None:
    """Return the station of the first ego footprint along the path meeting the target's sweep."""
    frames = np.arange(start, end)
    frames = frames[target.valid[frames]]
    if not len(frames):
        return None
    swept = union_all(shapely.polygons(box_corners(target.states[frames], target.size)))
    stations = np.arange(0.0, reach + CONFLICT_STEP_M, CONFLICT_STEP_M)
    position, heading, _ = path.frame(stations)
    poses = np.column_stack((position, heading))
    hits = shapely.intersects(shapely.polygons(box_corners(poses, scene.ego.size)), swept)
    return float(stations[np.argmax(hits)]) if hits.any() else None


def cached_profile(
    profiles: dict | None,
    v0: float,
    accel0: float,
    spec: Longitudinal,
    times: ArrayLike,
    max_speed: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``longitudinal_profile`` memoised in ``profiles`` (None: no cache).

    The returned arrays are read-only, since callers share them.
    """
    if profiles is None:
        return longitudinal_profile(v0, accel0, spec, times, max_speed)
    times = np.asarray(times, float)
    key = (
        float(v0).hex(),
        float(accel0).hex(),
        repr(spec),
        times.tobytes(),
        float(max_speed).hex(),
    )
    if key not in profiles:
        if len(profiles) >= PROFILE_CACHE_SIZE:
            profiles.clear()
        found = longitudinal_profile(v0, accel0, spec, times, max_speed)
        for array in found:
            array.setflags(write=False)
        profiles[key] = found
    return profiles[key]


def longitudinal_profile(
    v0: float, accel0: float, spec: Longitudinal, times: ArrayLike, max_speed: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return station, speed and acceleration at ``times`` (seconds from the start).

    The acceleration moves toward the plan's at its jerk, easing to zero at a standstill and
    before ``max_speed``; a recovery then switches to ``then_accel``.
    """
    times = np.asarray(times, float)
    h = MAX_INTEGRATION_STEP_S
    if len(times) > 1:
        h = min(MAX_INTEGRATION_STEP_S, float(np.diff(times).min()) / SUBSTEPS_PER_INTERVAL)
    goal, j = spec.accel, spec.jerk
    switched = spec.then_accel is None  # nothing to switch to
    stopped = None  # when the ego first stood still
    s, v, a = (
        0.0,
        max(v0, 0.0),
        float(np.clip(accel0, -VEHICLE_LIMITS.brake, VEHICLE_LIMITS.accel)),
    )
    out = np.empty((len(times), 3))
    clock, k = 0.0, 0
    while k < len(times):
        if clock >= times[k] - CLOCK_TOL_S:
            out[k] = s, v, a
            k += 1
            continue
        if not switched:
            due = _switch_due(spec, stopped)
            if due is not None and clock >= due - CLOCK_TOL_S:
                goal, j, switched = spec.then_accel, spec.then_jerk, True
        step = min(h, times[k] - clock)
        if not switched and due is not None and due - clock > CLOCK_TOL_S:
            step = min(step, due - clock)  # shrink the step to land exactly on the switch
        s, v, a = _advance(s, v, a, goal, j, step, max_speed)
        clock += step
        if v == 0.0 and stopped is None:
            stopped = clock
    return out[:, 0], out[:, 1], out[:, 2]


def _switch_due(spec: Longitudinal, stopped: float | None) -> float | None:
    """Return when the recovery switches: ``switch_s``, or ``wait_s`` after the ego stopped."""
    due = spec.switch_s if spec.wait_s is None else None
    if spec.wait_s is not None and stopped is not None:
        due = stopped + spec.wait_s
    return due


def _advance(
    s: float, v: float, a: float, goal: float, j: float, step: float, max_speed: float
) -> tuple[float, float, float]:
    """Return ``(s, v, a)`` one ``step`` on, with jerk up to ``j`` toward ``goal``."""
    if v <= STANDSTILL_MPS and a <= 0 and goal <= 0:  # stopped and not asked to start
        v, a, jerk = 0.0, 0.0, 0.0
    elif a < 0 and v <= a * a / (2 * j):  # release braking now so the stop is smooth
        jerk = j
    elif a > 0 and v >= max_speed - a * a / (2 * j):
        jerk = -j
    else:
        target = goal if goal < 0 or v < max_speed else 0.0
        jerk = min(max((target - a) / step, -j), j)  # np.clip on a scalar is slow
    ds = v * step + a * step**2 / 2 + jerk * step**3 / 6
    v_next = v + a * step + jerk * step**2 / 2
    if v_next < 0:  # the vehicle came to a stop inside this step
        v_next, ds = 0.0, max(ds, 0.0)
    s, v, a = s + ds, v_next, a + jerk * step
    if v == 0.0:
        a = 0.0
    return s, v, a


def lateral_profile(
    spec: Lateral, station: ArrayLike, length: float, lane_width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(d(s), d'(s))`` along the driven distance, using a quintic smoothstep."""
    offset = spec.offset_m * (lane_width if spec.in_lanes else 1.0)
    station = np.asarray(station, float)
    if offset == 0:
        return np.zeros_like(station), np.zeros_like(station)

    def step(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.clip(q, 0, 1)
        return 10 * q**3 - 15 * q**4 + 6 * q**5, 30 * q**2 - 60 * q**3 + 30 * q**4

    if spec.mode == "hold":
        value, slope = step(station / length)
        return offset * value, offset * slope / length
    half = length / 2
    up, up_slope = step(station / half)
    down, down_slope = step((station - half) / half)
    return offset * (up - down), offset * (up_slope - down_slope) / half


def _compose(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    station: np.ndarray,
    speed: np.ndarray,
    d: np.ndarray,
    slope: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray:
    """Combine a longitudinal profile and a lateral offset into world-frame desired states."""
    position, heading, curvature = frame
    tangent = heading_unit_vector(heading)
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    velocity = speed[:, None] * ((1 - d * curvature)[:, None] * tangent + slope[:, None] * normal)
    states = np.empty((len(station), STATE_SIZE))
    states[:, :2] = position + d[:, None] * normal
    moving = np.linalg.norm(velocity, axis=1) > VELOCITY_HEADING_MPS
    states[:, 2] = np.where(moving, np.arctan2(velocity[:, 1], velocity[:, 0]), heading)
    states[:, 3] = np.linalg.norm(velocity, axis=1)
    states[0] = initial
    return states


@dataclass(frozen=True, eq=False)
class Built:
    """Desired motions for the applicable candidates starting at the given frame.

    Attributes:
        rows: positions within the input candidates, applicable ones only.
        desired: ``[R, N, 4]`` desired states ``[x, y, heading, speed]``.
        commands: ``[R, N]`` steering commands; ``NaN`` where the path is tracked instead.
        stop_accel: per row, the deceleration solved for a stop at the conflict, or ``None``.
    """

    rows: tuple[int, ...]
    desired: np.ndarray
    commands: np.ndarray
    stop_accel: tuple
