"""Shared scene model, configuration, scene I/O and visibility.

Other submodules are imported by path, e.g. ``oncue.core.kinematics``.
"""

from .config import Config, DriverConfig, VisibilityConfig
from .io import (
    iter_scenes,
    load_config,
    load_scene,
    read_json,
    read_lines,
    save_scene,
    scene_inputs,
    write_json,
)
from .types import FeatureKind, MapFeature, RevealEvent, Scene, Track
from .visibility import reveal_events, target_visibility

__all__ = [
    "Config",
    "DriverConfig",
    "FeatureKind",
    "MapFeature",
    "RevealEvent",
    "Scene",
    "Track",
    "VisibilityConfig",
    "iter_scenes",
    "load_config",
    "load_scene",
    "read_json",
    "read_lines",
    "reveal_events",
    "save_scene",
    "scene_inputs",
    "target_visibility",
    "write_json",
]
