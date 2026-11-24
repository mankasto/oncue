"""`track_inputs` gives the model inputs for one (inserted) track at the scene's anchor, assembled
as at training time (`waymo_dataset`), plus its route. Generation and `compare` call this for every
target type."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ..utils.geometry import rotate
from .datasets.routes import route_chunks

if TYPE_CHECKING:  # annotations only
    from ..core.types import Scene, Track
    from .modeling.model import RouteModel

_SCENE: dict = {}


def track_inputs(
    scene: Scene, track: Track, model: RouteModel, route_xy: np.ndarray | None = None
) -> tuple[dict, np.ndarray, float]:
    """(batch of one sample, origin, heading): the scene encoded as in training
    (`waymo_dataset`), the track appended as one more road user (its history as valid in
    `track`), assembled in its frame (the same function as training), and `route_xy` (world
    polyline, or None: no route).
    The context sizes are the ones the model was trained with (`model.fcfg`)."""
    from .datasets.waymo.waymo_dataset import (
        AGENT_KEYS,
        assemble,
        collate,
        encode_agents,
        encode_scene,
        ground_points,
    )

    feature_cfg = getattr(model, "fcfg", None)
    if feature_cfg is None:
        raise ValueError("track_inputs: the feature settings (a loaded model's fcfg)")
    # keyed by what the encoding reads, not by the tracks dict: a standing target's start is
    # replace(scene, anchor=...), whose new dict holds the same tracks, so every start of a scene
    # shares one encoding (the encoding does not read the anchor)
    cache_key = (
        scene.scene_id,
        scene.ego_id,
        id(scene.map_features),
        id(scene.signals),
        len(scene.timestamps),
        tuple(map(id, scene.tracks.values())),
    )
    if cache_key not in _SCENE:  # the map part is the same for every candidate of a scene
        _SCENE.clear()
        # the scene kept with its encoding: while cached it cannot be freed, so no other scene
        # can take its id (a key of ids alone could match a new scene at a reused address)
        _SCENE[cache_key] = (encode_scene(scene), ground_points(scene), scene)
    record, ground, _ = _SCENE[cache_key]
    record = dict(record)
    own = encode_agents([track], scene.ego_id, ground)
    for key in AGENT_KEYS:
        record[key] = np.concatenate([record[key], own[key]])
    batch = collate([record])
    center = torch.tensor([len(record["agents"]) - 1])
    anchor = torch.tensor([scene.anchor])
    inputs = assemble(batch, center, anchor, feature_cfg)
    origin, h0 = track.states[scene.anchor, :2], track.states[scene.anchor, 2]
    route, usable = route_chunks(route_xy, origin, h0)
    inputs["route"] = torch.tensor(route, dtype=torch.float32)[None]
    inputs["route_ok"] = torch.tensor([float(usable)])
    inputs.pop("fut")
    return inputs, origin, h0


def to_world(local: np.ndarray, origin: np.ndarray, h0: float) -> np.ndarray:
    """Agent-frame positions [..., 2] back to the world frame."""
    return rotate(local, h0) + origin


def to_local(points: np.ndarray, origin: np.ndarray, h0: float) -> np.ndarray:
    """World positions [..., 2] to the agent frame."""
    return rotate(np.asarray(points, float) - origin, -h0)
