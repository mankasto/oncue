"""Per-scene state shared by every generation step, built once per scene by `scene_context`."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import shapely

from ..core.config import Config, DriverConfig
from ..core.rules import lane_speed_limit, logged_path, traffic_rules
from ..core.types import FeatureKind, Scene
from ..core.visibility import static_region
from ..data.map_geometry import drivable_region
from ..oracle.evaluate import Oracle
from ..utils.geometry import MIN_LINE_POINTS, union_all
from .checks.validity import road_user_contacts
from .reference import protocol_ego


def scene_context(scene: Scene, config: Config, driver: DriverConfig) -> SceneContext:
    limit = lane_speed_limit(scene)
    reference, _ = protocol_ego(scene, None, driver, speed_limit=limit)
    path = logged_path(scene)
    ego, anchor = reference.ego, scene.anchor
    sim = np.flatnonzero(ego.valid)
    sim = sim[sim >= anchor]
    front = np.array([path.project(shapely.Point(*ego.states[i, :2])) for i in sim])
    front = front + ego.size[0] / 2
    lines = [
        shapely.LineString(f.points)
        for f in scene.map_features
        if f.kind is FeatureKind.ROAD_EDGE and len(f.points) >= MIN_LINE_POINTS
    ]
    road = drivable_region(scene)
    edge_lines = union_all(lines) if lines else None
    if edge_lines is not None:
        shapely.prepare(edge_lines)  # candidates' footprints are tested against it many times
    return SceneContext(
        scene=scene,
        reference=reference,
        config=config,
        driver=driver,
        path=path,
        limit=limit,
        sim=sim,
        front=front,
        static=static_region(scene, config.visibility),
        road=road,
        edge_lines=edge_lines,
        oracle=Oracle(config.oracle),
        reference_contact=first_reference_contact(reference, config),
    )


def first_reference_contact(reference: Scene, config: Config) -> tuple[str, int] | None:
    """Return (road user, frame) of the reference ego's first background contact, or None."""
    contacts = road_user_contacts(
        reference, reference.ego_id, config.oracle.collision_substeps, config.replay.still_mps
    )
    return None if not contacts else (contacts[0][0], contacts[0][1])


@dataclass
class SceneContext:
    scene: Scene  # the logged scene
    reference: Scene  # same scene with the ego replaced by the protocol reference ego
    config: Config
    # shared driver settings; the reference ego adds its own (`reference_settings`)
    driver: DriverConfig
    path: shapely.LineString  # ego's logged path from the anchor on
    limit: float  # lane speed limit (m/s)
    sim: np.ndarray  # frames the reference ego is simulated (>= anchor)
    front: np.ndarray  # its front's station along `path` at those frames
    static: shapely.Geometry | None  # static occluders (visibility)
    road: shapely.Geometry  # drivable region
    edge_lines: shapely.Geometry | None  # union of the road-edge lines
    oracle: Oracle
    reference_contact: tuple[str, int] | None  # from `first_reference_contact`
    # funnel counts per stage (route, contact, start); drops keyed "stage/reason"
    considered: Counter = field(default_factory=Counter)
    drops: Counter = field(default_factory=Counter)

    def consider(self, stage: str, n: int = 1) -> None:
        self.considered[stage] += n

    def drop(self, stage: str, reason: str, n: int = 1) -> None:
        self.drops[f"{stage}/{reason}"] += n

    @property
    def anchor(self) -> int:
        return self.scene.anchor

    @property
    def dt(self) -> float:
        return self.scene.dt

    @property
    def n(self) -> int:
        return len(self.scene.timestamps)

    def arrival(self, station: float) -> float | None:
        """Return the absolute time (s) the reference ego's front reaches a station, or None."""
        reach = np.flatnonzero(self.front >= station)
        return None if not len(reach) else float(self.sim[reach[0]] * self.dt)

    def reference_schedule(self) -> ReferenceSchedule:
        """Return the reference ego's clock, to time triggered starts and measure lateness."""
        return ReferenceSchedule(
            stations=np.maximum.accumulate(self.front),
            frames=self.sim.astype(float),
            speeds=self.reference.ego.states[self.sim, 3],
        )

    def signal_stations(self) -> list[float]:
        """Return the stations of signals along the ego's path, for signalised-route detection."""
        return [st for st, _ in traffic_rules(self.scene, self.limit, self.driver).signals]


class ReferenceSchedule(NamedTuple):
    """The reference ego's progress over time (``SceneContext.reference_schedule``)."""

    stations: np.ndarray  # front's station along the path, never decreasing
    frames: np.ndarray  # frames it is simulated at (as floats)
    speeds: np.ndarray  # its speed at those frames
