"""Where a hidden target crosses the ego's path (`routes`): crossing routes at the sites beside
a stationary hiding vehicle, and lane routes along map lanes for vehicles."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import shapely

from ...core.config import SiteConfig
from ...core.road import RoadEdges
from ...core.rules import chain_stop_lines, logged_path, stop_sign_lanes
from ...core.types import FeatureKind, RoadEdgeType, Scene
from ...data.map_geometry import (
    OFFSET_CHORD_M,
    chain_speed_limit,
    lane_crossings,
    lane_meeting_points,
    path_offset,
)
from ...utils.geometry import (
    MIN_DIVISOR,
    MIN_LINE_POINTS,
    box_corners,
    cumulative_arc_length,
    heading_unit_vector,
    line_heading,
    union_all,
    wrap_angle,
)
from ...utils.tracks import frame_of
from ..context import SceneContext

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Collection

    from ...core.types import Track
    from ...data.map_geometry import LaneCrossing
    from ..targets import TargetSpec

CROSSING_SITES = ("kerb", "median", "lot", "queue")  # sites beside a hiding vehicle
CONTACT_STATION_M = 10.0  # a contact point lies this near the crossing along the ego path
STOP_HOLD = (3.0, 25.0)  # a stop line this far before the crossing: the vehicle waits there
STOP_LINE_GAP_M = 0.5  # a waiting vehicle's front stops this far before its stop line
# chord either side when taking a route's heading
ROUTE_HEADING_CHORD_M = 0.25
LATERAL_SEARCH_M = (40.0, 20.0)  # s_at_lateral searches this far before and after the crossing
# a parked or queued vehicle must be observed this many frames from the anchor
STATIONARY_MIN_FRAMES = 30
STATIONARY_MPS = 0.5  # and slower than this in `SiteConfig.stationary_share` of them
PATH_END_MARGIN_M = 2.0  # a hiding vehicle must be at least this far before the logged path's end
QUEUE_ALIGN_DEG = 30.0  # a queued vehicle must run along the path (either way) within this
CENTRE_ONLY = (0.0, 0.0)  # a box of no size: `RoadEdges.depth` of its centre
WALK_STEP_M = 0.5  # a crossing route's walkable stretch is tested every this far
# a lane route is "signalised" if an ego-path signal sits within this far before it
SIGNALISED_BEFORE_M = 40
LANE_ROUTE_MIN_PATH_M = 20  # lane routes need an ego path at least this long
STRAIGHT_ACROSS_DEG = 90.0  # square to the ego's direction of travel
LATERAL_SEARCH_STEP_M = 0.05  # s_at_lateral samples the route this often
EXACT_OFFSET_M = 1e-6  # sampled offsets this near the one asked for are recomputed exactly
SIDE_BEFORE_M = 5  # a lane route's side of the path is read this far before the crossing
MIN_SEARCH_SAMPLES = 2  # s_at_lateral needs at least this many samples to look for a sign change
END_SLACK_M = 1e-9  # np.arange stops this far past its end so a sample at the end is kept


def routes(ctx: SceneContext, spec: TargetSpec) -> list[Route]:
    """Return the spec's routes: crossing routes at its hiding sites, then its lane routes."""
    out = []
    if set(spec.sites) & set(CROSSING_SITES):
        out += crossing_routes(ctx, spec)
    if "lane" in spec.sites:
        out += lane_routes(ctx, spec.size)
    return out


def has_sites(scene: Scene, kinds: Collection[str]) -> bool:
    """Return whether the scene has a site of any of `kinds`, from the logged path alone."""
    crossing = [k for k in kinds if k in CROSSING_SITES]
    if crossing and hiding_sites(scene, crossing, SiteConfig()):
        return True
    if "lane" in kinds:
        path = logged_path(scene)
        if path.length < LANE_ROUTE_MIN_PATH_M:
            return False
        lanes = [
            f
            for f in scene.map_features
            if f.kind is FeatureKind.LANE and len(f.points) >= MIN_LINE_POINTS
        ]
        return next(lane_meeting_points(lanes, path), None) is not None
    return False


# ---------------------------------------------------------------- hiding sites
@dataclass(frozen=True)
class Site:
    car_id: str  # vehicle hiding the target
    station: float  # where the target crosses, along the ego's logged path
    lateral: float  # signed offset from the path where the target stands (left positive)
    kind: str = "kerb"  # kerb | median | lot (beyond a hiding vehicle) | queue (in a queue's gap)
    other_id: str | None = None  # queue: the vehicle in front of the gap


class _Car(NamedTuple):
    """A stationary vehicle beside the ego's logged path ahead."""

    track_id: str
    track: Track
    state: np.ndarray  # at the anchor
    station: float  # along the path
    lateral: float  # signed offset from the path, left positive


def hiding_sites(scene: Scene, kinds: Collection[str], cfg: SiteConfig) -> list[Site]:
    """Return sites of `kinds` beside the ego's logged path ahead."""
    path = logged_path(scene)
    cars = _stationary_beside(scene, path, cfg)
    if not cars:
        return []
    kind_of = _CarKinds(scene, path, cfg)
    sites, queued = [], []
    for car in cars:
        kind = kind_of(car)
        if kind == "queue":
            queued.append(car)
        elif kind in kinds and _within(abs(car.lateral), _lateral_range(kind, cfg)):
            sites.append(_front_site(car, kind, cfg))
    if "queue" in kinds:
        sites += queue_gaps(path, queued, cfg)
    return sites


def _stationary_beside(scene: Scene, path: shapely.LineString, cfg: SiteConfig) -> list[_Car]:
    """Return stationary vehicles beside the path ahead, as far out as any site kind looks."""
    anchor = scene.anchor
    ranges = (cfg.lateral_range_m, cfg.lot_lateral_m, cfg.queue_lateral_m)
    reach = (min(r[0] for r in ranges), max(r[1] for r in ranges))
    cars = []
    for k, t in scene.tracks.items():
        if k == scene.ego_id or t.agent_type != "vehicle" or not t.valid[anchor]:
            continue
        if not _stationary(scene, t, cfg):
            continue
        station, lat = path_offset(path, t.states[anchor, :2])
        ahead = cfg.min_ahead_m <= station <= path.length - PATH_END_MARGIN_M
        if ahead and _within(abs(lat), reach):
            cars.append(_Car(k, t, t.states[anchor], station, lat))
    return cars


class _CarKinds:
    """Classify a stationary vehicle beside the path: lot (off the road), kerb or median (by the
    road edge beside it), else queue."""

    def __init__(self, scene: Scene, path: shapely.LineString, cfg: SiteConfig):
        self.path, self.cfg, self.anchor = path, cfg, scene.anchor
        self.road = RoadEdges(scene)
        self.edges = [
            f
            for f in scene.map_features
            if f.kind is FeatureKind.ROAD_EDGE and len(f.points) >= MIN_LINE_POINTS
        ]
        lines = [shapely.LineString(f.points) for f in self.edges]
        self.tree = shapely.STRtree(lines) if lines else None

    def __call__(self, car: _Car) -> str:
        if self.road.depth(car.state[None], CENTRE_ONLY)[0] > 0:  # NaN (unknown) is not off
            return "lot"
        if self.tree is None:
            return "queue"
        corners = box_corners(car.state[None], car.track.size)[0]
        outer = shapely.Point(*max(corners, key=lambda c: abs(path_offset(self.path, c)[1])))
        nearest = int(self.tree.query_nearest(outer)[0])
        if self.tree.geometries[nearest].distance(outer) > self.cfg.kerb_edge_m:
            return "queue"
        return "median" if self.edges[nearest].edge_type == RoadEdgeType.MEDIAN else "kerb"


def _front_site(car: _Car, kind: str, cfg: SiteConfig) -> Site:
    """Return the site sitting just past a hiding vehicle's front, beyond its outer side."""
    length, width = car.track.size[0], car.track.size[1]
    side = np.sign(car.lateral)
    stand = side * (abs(car.lateral) + width / 2 + cfg.stand_beyond_m)
    return Site(car.track_id, car.station + length / 2 + cfg.gap_beyond_m, stand, kind)


def _lateral_range(kind: str, cfg: SiteConfig) -> tuple[float, float]:
    return cfg.lot_lateral_m if kind == "lot" else cfg.lateral_range_m


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]


def queue_gaps(path: shapely.LineString, cars: list[_Car], cfg: SiteConfig) -> list[Site]:
    """Return queue sites: gaps between two consecutive vehicles queued in one traffic lane."""
    lo, hi = cfg.queue_lateral_m
    lined = [c for c in cars if lo <= abs(c.lateral) <= hi and _along_path(path, c)]
    sites = []
    for side in (-1, 1):
        row = sorted((c for c in lined if np.sign(c.lateral) == side), key=lambda c: c.station)
        for i, behind in enumerate(row):
            # next car ahead in the same lane
            tolerance = cfg.queue_lane_tolerance_m
            same_lane = (c for c in row[i + 1 :] if abs(c.lateral - behind.lateral) <= tolerance)
            front = next(same_lane, None)
            if front is None:
                continue
            behind_end = behind.station + behind.track.size[0] / 2
            gap = (front.station - front.track.size[0] / 2) - behind_end
            if cfg.queue_gap_m[0] <= gap <= cfg.queue_gap_m[1]:
                lateral = (behind.lateral + front.lateral) / 2
                sites.append(
                    Site(behind.track_id, behind_end + gap / 2, lateral, "queue", front.track_id)
                )
    return sites


def _along_path(path: shapely.LineString, car: _Car) -> bool:
    _, tangent, _ = path_frame(path, np.array([car.station]))
    off = wrap_angle(car.state[2] - np.arctan2(tangent[0, 1], tangent[0, 0]))
    return abs(np.cos(off)) >= np.cos(np.radians(QUEUE_ALIGN_DEG))


def _stationary(scene: Scene, t: Track, cfg: SiteConfig) -> bool:
    anchor = scene.anchor
    seen = np.flatnonzero(t.valid)
    seen = seen[seen >= anchor]
    return (
        len(seen) >= STATIONARY_MIN_FRAMES
        and np.mean(t.states[seen, 3] < STATIONARY_MPS) >= cfg.stationary_share
    )


def stands_still(track: Track, first: int, last: int, drift_m: float) -> bool:
    """Return whether a road user is valid from `first` to `last` and stays within `drift_m`."""
    frames = slice(first, last + 1)
    if not track.valid[frames].all():
        return False
    xy = track.states[frames, :2]
    return bool(np.hypot(*(xy - xy[0]).T).max() <= drift_m)


def path_frame(path: shapely.LineString, stations: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return ([K, 2] points, unit tangents, unit left normals) of the path at `stations`."""
    p = shapely.get_coordinates(shapely.line_interpolate_point(path, stations))
    ahead = np.minimum(stations + OFFSET_CHORD_M, path.length)
    q = shapely.get_coordinates(shapely.line_interpolate_point(path, ahead))
    tangent = (q - p) / np.maximum(np.hypot(*(q - p).T), MIN_DIVISOR)[:, None]
    return p, tangent, np.column_stack([-tangent[:, 1], tangent[:, 0]])


# ---------------------------------------------------------------- crossing routes
def crossing_routes(ctx: SceneContext, spec: TargetSpec) -> list[Route]:
    """Return the spec's crossing routes at each of its hiding sites (only straight at a queue)."""
    cfg = SiteConfig()
    sites = hiding_sites(ctx.scene, [k for k in spec.sites if k in CROSSING_SITES], cfg)
    obstacles = Obstacles(ctx, spec.size, cfg)
    out = []
    for site in sites:
        for shape in route_shapes(spec):
            if site.kind == "queue" and shape[0] != "straight":
                continue
            ctx.consider("route")
            route, reason = crossing_route(ctx, spec, site, shape, obstacles, cfg)
            if route is None:
                ctx.drop("route", reason)
            else:
                out.append(route)
    return out


def route_shapes(spec: TargetSpec) -> list[tuple[str, float]]:
    """Return (route kind, crossing angle) for each crossing route a site gets."""
    shapes = []
    if "straight" in spec.route_kinds:
        shapes += [("straight", float(a)) for a in spec.crossing_angles_deg]
    if "sidewalk" in spec.route_kinds:
        shapes.append(("sidewalk", STRAIGHT_ACROSS_DEG))
    if "oblique" in spec.route_kinds:
        shapes += [("oblique", float(a)) for a in spec.oblique_angles_deg]
    return shapes


def crossing_route(
    ctx: SceneContext,
    spec: TargetSpec,
    site: Site,
    shape: tuple[str, float],
    obstacles: Obstacles,
    cfg: SiteConfig,
) -> tuple[Route | None, str | None]:
    """Return (route, None) or (None, reason left out) for a route of `shape` at `site`.

    The route runs from a walkable lead-in to the hiding spot, then across the ego's path.
    """
    route_kind, angle = shape
    p, tangent, normal = (a[0] for a in path_frame(ctx.path, np.array([site.station])))
    side = float(np.sign(site.lateral))
    spot = p + normal * site.lateral  # hiding spot, beyond the hider's outer side
    a = np.radians(angle)
    across = np.cos(a) * tangent - side * np.sin(a) * normal  # towards the far side
    end = spot + across * (abs(site.lateral) + cfg.cross_to_m) / np.sin(a)
    meeting = _meeting(ctx.path, spot, end, site.station)
    if meeting is None:
        return None, "crossing_off_path"
    station, to_path = meeting
    t_ego = ctx.arrival(station)
    if t_ego is None:
        return None, "ego_never_arrives"
    if not _hider_stands(ctx, spec, site, t_ego, cfg):
        return None, "hider_moves"
    ahead = spot + np.arange(0.0, to_path + END_SLACK_M, WALK_STEP_M)[:, None] * across
    if obstacles.blocked(ahead, np.full(len(ahead), np.arctan2(across[1], across[0]))).any():
        return None, "crossing_blocked"
    length = lead_in_length(ctx, spec, t_ego, cfg)
    lead = lead_in(ctx.path, site, route_kind, spot, across, length)
    lead = lead[obstacles.walkable_from(lead) :]
    s_spot = float(cumulative_arc_length(lead)[-1])
    route = Route(
        kind=site.kind,
        line=shapely.LineString(np.vstack([lead, end])),
        conflict_station=station,
        t_ego=t_ego,
        side=side,
        s_centre=s_spot + to_path,
        angle_deg=float(angle),
        route_kind=route_kind,
        site_id=site.car_id,
        other_id=site.other_id,
        meta={"s_stand": s_spot},
    )
    return route, None


def _meeting(
    path: shapely.LineString, spot: np.ndarray, end: np.ndarray, near: float
) -> tuple[float, float] | None:
    """Return (station, distance from `spot`) where segment spot-end meets `path` nearest `near`,
    or None."""
    meet = shapely.LineString([spot, end]).intersection(path)
    points = [
        g for g in getattr(meet, "geoms", [meet]) if g.geom_type == "Point" and not g.is_empty
    ]
    if not points:
        return None
    x = min(points, key=lambda g: abs(path.project(g) - near))
    return float(path.project(x)), float(np.linalg.norm(np.array(x.coords[0]) - spot))


def _hider_stands(
    ctx: SceneContext, spec: TargetSpec, site: Site, t_ego: float, cfg: SiteConfig
) -> bool:
    """Return whether the site's vehicles stand still until shortly after the reference ego
    crosses."""
    first = spec.first_frame(ctx.anchor, ctx.dt)
    last = min(frame_of(t_ego + cfg.still_after_contact_s, ctx.dt), ctx.n - 1)
    hiders = [site.car_id] if site.other_id is None else [site.car_id, site.other_id]
    return all(stands_still(ctx.scene.tracks[k], first, last, cfg.still_drift_m) for k in hiders)


def lead_in_length(ctx: SceneContext, spec: TargetSpec, t_ego: float, cfg: SiteConfig) -> float:
    """Return how far a route runs back from the hiding spot, so every moving start fits."""
    duration = t_ego - ctx.anchor * ctx.dt + spec.assumed_history_s
    speed = spec.log.speed_median_mps
    if spec.start_rule == "transition" and spec.log.start_speeds_mps:
        speed = max(speed, *spec.log.start_speeds_mps)
    return cfg.lead_in_speed_factor * speed * duration


def lead_in(
    path: shapely.LineString,
    site: Site,
    route_kind: str,
    spot: np.ndarray,
    across: np.ndarray,
    length: float,
) -> np.ndarray:
    """Return [K, 2] points every WALK_STEP_M over `length` ending at the hiding `spot`."""
    back = np.arange(0.0, length + END_SLACK_M, WALK_STEP_M)[::-1]
    if route_kind == "straight":
        return spot - back[:, None] * across
    p, _, normal = path_frame(path, site.station - back[site.station - back >= 0])
    return p + site.lateral * normal  # its last point: the spot


class Obstacles:
    """Static objects and standing road users a crossing route may not pass through."""

    def __init__(self, ctx: SceneContext, size: tuple[float, float], cfg: SiteConfig):
        scene, anchor = ctx.scene, ctx.anchor
        boxes = [
            shapely.Polygon(box_corners(t.states[anchor][None], t.size)[0])
            for k, t in scene.tracks.items()
            if k != scene.ego_id and t.valid[anchor] and _stationary(scene, t, cfg)
        ]
        standing = union_all(boxes) if boxes else None
        self.parts = [g for g in (ctx.static, standing) if g is not None]
        for g in self.parts:
            shapely.prepare(g)
        self.size = size

    def blocked(self, points: np.ndarray, headings: np.ndarray) -> np.ndarray:
        """Return [K] whether the target's box at each point and heading touches an obstacle."""
        states = np.column_stack([points, headings])
        boxes = shapely.polygons(box_corners(states, self.size))
        out = np.zeros(len(points), bool)
        for g in self.parts:
            out |= shapely.intersects(g, boxes)
        return out

    def walkable_from(self, points: np.ndarray) -> int:
        """Return the first index from which the target walks over free ground, facing forward."""
        if len(points) < MIN_LINE_POINTS:
            return 0
        step = np.diff(points, axis=0)
        headings = np.arctan2(step[:, 1], step[:, 0])
        hits = np.flatnonzero(self.blocked(points[:-1], headings))
        return int(hits[-1]) + 1 if len(hits) else 0


@dataclass
class Route:
    kind: str  # site's kind: kerb | median | lot | queue | lane
    line: shapely.LineString  # target's path in the world frame, from where it starts
    conflict_station: float  # station of the crossing along the ego's logged path
    t_ego: float  # absolute time the reference ego's front reaches conflict_station
    side: float  # +1 / -1: side of the ego path the target comes from (left positive)
    s_centre: float  # arc along `line` where the target's centre is on the ego path
    angle_deg: float  # crossing angle with the ego path
    route_kind: str  # straight | sidewalk | oblique | lane
    site_id: str | None = None  # the vehicle hiding it (crossing routes)
    other_id: str | None = None  # queue: the vehicle in front of the gap
    s_hold: float | None = None  # lane: the stop line it waits at (stop-controlled), else None
    meta: dict = field(default_factory=dict)

    def point(self, s: float) -> np.ndarray:
        return np.array(self.line.interpolate(float(s)).coords[0])

    def heading(self, s: float) -> float:
        chord = ROUTE_HEADING_CHORD_M
        return line_heading(self.line, s, back=chord, ahead=chord)

    def s_at_lateral(
        self, path: shapely.LineString, lateral: float, ahead_m: float = 0.0
    ) -> float | None:
        """Return the arc nearest the crossing where the point `ahead_m` in front of the target's
        centre has offset `lateral` from the ego path, or None."""
        before, after = LATERAL_SEARCH_M
        lo, hi = max(self.s_centre - before, 0.0), min(self.s_centre + after, self.line.length)
        s = np.arange(lo, hi, LATERAL_SEARCH_STEP_M)
        if len(s) < MIN_SEARCH_SAMPLES:
            return None
        # vectorized `point` and `heading`
        pts = shapely.get_coordinates(shapely.line_interpolate_point(self.line, s))
        if ahead_m:
            chord = ROUTE_HEADING_CHORD_M
            h = line_heading(self.line, s, back=chord, ahead=chord)
            pts = pts + ahead_m * heading_unit_vector(h)
        # vectorized offsets find the sign changes; the deciding ones are redone exactly
        d = _laterals(path, pts) - lateral
        exact = {}

        def at(j: int) -> float:
            if j not in exact:
                exact[j] = path_offset(path, pts[j])[1] - lateral
            return exact[j]

        for j in np.flatnonzero(np.abs(d) < EXACT_OFFSET_M):
            d[j] = at(int(j))
        cross = np.flatnonzero(np.sign(d[:-1]) != np.sign(d[1:]))
        if not len(cross):
            return None
        i = int(cross[np.argmin(np.abs(s[cross] - self.s_centre))])
        di, dj = at(i), at(i + 1)
        arc = float(s[i] + (s[i + 1] - s[i]) * di / (di - dj))
        if abs(path_offset(path, pts[i])[0] - self.conflict_station) > CONTACT_STATION_M:
            return None  # the offset is reached by another crossing, not this one
        return arc


def _laterals(path: shapely.LineString, xy: np.ndarray) -> np.ndarray:
    """Return [N] signed lateral offsets: `path_offset` vectorized, up to the last bits."""
    stations = shapely.line_locate_point(path, shapely.points(xy))
    p, tangent, _ = path_frame(path, stations)
    r = xy - p
    return tangent[:, 0] * r[:, 1] - tangent[:, 1] * r[:, 0]


# ---------------------------------------------------------------- lane routes
def lane_routes(ctx: SceneContext, size: tuple[float, float]) -> list[Route]:
    """Return a route per map lane chain crossing the ego's logged path ahead at 45-135 degrees."""
    scene, path = ctx.scene, ctx.path
    if path.length < LANE_ROUTE_MIN_PATH_M:
        return []
    lanes = {
        f.feature_id: f
        for f in scene.map_features
        if f.kind is FeatureKind.LANE and len(f.points) >= MIN_LINE_POINTS
    }
    stop_lanes = stop_sign_lanes(scene)
    signals = ctx.signal_stations()
    out = []
    for c, why in lane_crossings(lanes, path, scene.ego):
        ctx.consider("route")
        if why is not None:
            ctx.drop("route", why)
            continue
        t_ego = ctx.arrival(c.cross_station)
        if t_ego is None:
            ctx.drop("route", "ego_never_arrives")
            continue
        held = _stop_line_hold(c, size, stop_lanes, lanes)
        if held is None:
            ctx.drop("route", "stop_line_at_crossing")
            continue
        pts, owner, line, lane_station, s_hold = held
        out.append(
            Route(
                kind="lane",
                line=line,
                conflict_station=float(c.cross_station),
                t_ego=t_ego,
                side=_side_of(path, line, lane_station),
                s_centre=float(lane_station),
                angle_deg=float(c.angle_deg),
                route_kind="lane",
                s_hold=s_hold,
                meta=_lane_meta(c, lanes, pts, owner, signals),
            )
        )
    return out


def _stop_line_hold(c: LaneCrossing, size: tuple[float, float], stop_lanes: set, lanes: dict):
    """Return ``(points, owner, line, lane_station, s_hold)`` of a crossing's route, ``s_hold``
    where a stop line holds the target (None: none does); None when the stop line sits at the
    crossing itself. A stop line far before the crossing starts the route there instead."""
    pts, owner, line, lane_station = c.points, c.owner, c.line, c.lane_station
    stops = [
        at
        for at in chain_stop_lines(pts, owner, c.chain, stop_lanes, lanes)
        if at <= lane_station + 1
    ]
    if not stops:
        return pts, owner, line, lane_station, None
    s_hold = stops[-1] - size[0] / 2 - STOP_LINE_GAP_M
    if lane_station - s_hold > STOP_HOLD[1]:
        pts, owner = _from_arc(pts, owner, cumulative_arc_length(pts), stops[-1])
        line = shapely.LineString(pts)
        return pts, owner, line, line.project(c.point), None
    if lane_station - s_hold < STOP_HOLD[0]:
        return None
    return pts, owner, line, lane_station, s_hold


def _side_of(path: shapely.LineString, line: shapely.LineString, lane_station: float) -> float:
    """Return the side of ``path`` the lane comes from (+1 left, -1 right; +1 on the path),
    read ``SIDE_BEFORE_M`` before the crossing."""
    before = np.array(line.interpolate(max(lane_station - SIDE_BEFORE_M, 0)).coords[0])
    return float(np.sign(path_offset(path, before)[1]) or 1.0)


def _lane_meta(c: LaneCrossing, lanes: dict, pts: np.ndarray, owner: list, signals: list) -> dict:
    """Return what a lane route records about its lane chain."""
    lane_id, chain = c.lane.feature_id, c.chain
    return {
        "lane": lane_id,
        "limit": chain_speed_limit(lanes, chain, lane_id),
        # highest posted limit up to the crossing lane
        "limit_max": max(
            (
                lanes[i].speed_limit_mps
                for i in chain[: chain.index(lane_id) + 1]
                if lanes[i].speed_limit_mps
            ),
            default=None,
        ),
        "chain": chain,
        "owner": owner,
        "points": pts,
        "signalised": any(
            c.cross_station - SIGNALISED_BEFORE_M <= st <= c.cross_station for st in signals
        ),
        "level_gap_m": c.level_gap_m,  # None when the elevation is not recorded
    }


def _from_arc(pts: np.ndarray, owner: list[str], arcs: np.ndarray, cut: float):
    """Return (points, owner lane per point) of a lane route from arc `cut` onward."""
    k = int(np.searchsorted(arcs, cut, side="right"))  # first point beyond the cut
    w = (cut - arcs[k - 1]) / max(arcs[k] - arcs[k - 1], MIN_DIVISOR)
    first = pts[k - 1] + w * (pts[k] - pts[k - 1])
    return np.vstack([first, pts[k:]]), [owner[k - 1], *owner[k:]]
