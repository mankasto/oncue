"""Per-scene inputs the hard checks and flaws read beyond `SceneContext`, built once per scene
by `scene_rules`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import shapely

from ...core.cutoff import route_band
from ...core.road import LaneDirections, RoadEdges, VehicleLanes
from ...core.rules import driver_path, stop_sign_lanes, traffic_rules
from ...core.types import FeatureKind

if TYPE_CHECKING:  # import only for type hints
    import numpy as np

    from ..context import SceneContext


@dataclass(frozen=True, eq=False)
class SceneRules:
    """One scene's shared inputs for the checks and flaws."""

    band: shapely.Geometry  # reference ego's route band (`core.cutoff.route_band`)
    # (stop-line station, phase per frame) for each signal along the ego's path
    ego_signals: tuple[tuple[float, np.ndarray], ...]
    stop_lanes: set[str]  # lanes controlled by stop signs
    lanes: dict  # lane id -> its map feature
    directions: LaneDirections
    vehicle_lanes: VehicleLanes  # with posted limits, used by the speeding flaw
    edges: RoadEdges


_LAST: list = []  # [(scene context, its SceneRules)]: the last scene's, cached for its candidates


def scene_rules(ctx: SceneContext) -> SceneRules:
    """Return `ctx`'s SceneRules, cached for the latest scene only."""
    if not _LAST or _LAST[0][0] is not ctx:
        _LAST[:] = [(ctx, _build(ctx))]
    return _LAST[0][1]


def _build(ctx: SceneContext) -> SceneRules:
    scene = ctx.scene  # ctx.static comes prepared (``static_region``)
    route = driver_path(scene, ctx.driver)
    return SceneRules(
        band=route_band(shapely.get_coordinates(route), ctx.reference.ego.size[1]),
        ego_signals=traffic_rules(scene, ctx.limit, ctx.driver).signals,
        stop_lanes=stop_sign_lanes(scene),
        lanes={f.feature_id: f for f in scene.map_features if f.kind is FeatureKind.LANE},
        directions=LaneDirections(scene),
        vehicle_lanes=VehicleLanes(scene),
        edges=RoadEdges(scene),
    )
