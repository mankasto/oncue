"""Scene data shared across the site's pages: rounding, map features, and tracks near a region."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.types import FeatureKind, RoadEdgeType, RoadLineType, Scene

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable

    import numpy.typing as npt
    import shapely

MAP_KINDS = (FeatureKind.LANE, FeatureKind.ROAD_LINE, FeatureKind.ROAD_EDGE, FeatureKind.CROSSWALK)


def rounded(values: npt.ArrayLike, digits: int = 2) -> list:
    """Return ``values`` as a nested list rounded to ``digits``, matching the pages' format."""
    return np.round(np.asarray(values, float), digits).tolist()


def focus_box(points: npt.ArrayLike, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(low, high)`` corners of ``points``' box widened by ``margin``."""
    pts = np.asarray(points, float).reshape(-1, 2)
    return pts.min(axis=0) - margin, pts.max(axis=0) + margin


def near_box(points: npt.ArrayLike, margin: float) -> Callable[[npt.ArrayLike], bool]:
    """Return a predicate testing whether a point set comes within ``margin`` of ``points``' box."""
    low, high = focus_box(points, margin)

    def near(other: npt.ArrayLike) -> bool:
        p = np.asarray(other).reshape(-1, 2)
        return bool(((p >= low) & (p <= high)).all(axis=1).any())

    return near


STATIC_SIMPLIFY_M = 0.5  # simplification tolerance for drawn walls, to keep the page small


def static_payload(
    static: shapely.Geometry | None, focus: npt.ArrayLike, margin: float
) -> list | None:
    """Return the static occluders within ``margin`` of ``focus`` as simplified polygons.

    Each polygon is ``[outer_ring, hole_ring, ...]`` of scene-frame ``[x, y]`` points; None when
    nothing is near.
    """
    import shapely

    if static is None or static.is_empty:
        return None
    (low_x, low_y), (high_x, high_y) = focus_box(focus, margin)
    clipped = shapely.clip_by_rect(static, low_x, low_y, high_x, high_y).simplify(STATIC_SIMPLIFY_M)
    polygons = [
        part
        for part in shapely.get_parts(clipped)
        if part.geom_type == "Polygon" and not part.is_empty
    ]
    rings = [
        [rounded(np.asarray(p.exterior.coords)[:, :2], 1)]
        + [rounded(np.asarray(hole.coords)[:, :2], 1) for hole in p.interiors]
        for p in polygons
    ]
    return rings or None


def map_payload(scene: Scene, near: Callable[[npt.ArrayLike], bool]) -> list[dict]:
    return [
        {
            "kind": f.kind.value,
            "points": rounded(f.points, 1),
            "median": f.edge_type is RoadEdgeType.MEDIAN,
            "paint": line_paint(f.line_type),
        }
        for f in scene.map_features
        if f.kind in MAP_KINDS and near(f.points)
    ]


def line_paint(line_type: RoadLineType | int | None) -> str | None:
    """Return a road line's paint colour, ``"yellow"`` or ``"white"``, or None when unknown."""
    if line_type is None:
        return None
    name = RoadLineType(line_type).name
    return "yellow" if "YELLOW" in name else "white" if "WHITE" in name else None


def tracks_payload(scene: Scene, near: Callable[[npt.ArrayLike], bool]) -> dict:
    """Return the road users other than the ego that come near the case, by track id."""
    out = {}
    for key, track in scene.tracks.items():
        if key == scene.ego_id:
            continue
        observed = track.states[track.valid, :2]
        if not (len(observed) and near(observed)):
            continue
        out[key] = {
            "type": track.agent_type,
            "size": rounded(track.size),
            "xyh": [
                row if ok else None
                for row, ok in zip(rounded(track.states[:, :3]), track.valid, strict=True)
            ],
        }
    return out
