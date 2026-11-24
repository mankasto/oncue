"""Case viewer: OnCue tests replayed in the closed loop, as an offline HTML page.

Start from ``render_cases`` (``oncue cases``).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import replace
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import DriverConfig
from ..core.io import read_lines, scene_from_dict, scene_inputs
from ..core.replay import fill_gaps
from ..core.types import Track
from ..core.visibility import target_visibility
from ..utils.parallel import run_jobs
from ..utils.tracks import frame_of
from .payload import map_payload, near_box, rounded, static_payload, tracks_payload

if TYPE_CHECKING:  # annotations only
    from collections.abc import Sequence

CASE_PLANNERS = ("reactive", "conservative", "occlusion_aware")
CASE_VIEW_MARGIN_M = 50.0  # a case page draws the map this far around the target and the ego
KEY_FIELDS = ("site", "route", "profile", "contact_key", "W_req", "generator")
CASES_PER_KIND = 4  # cases per target type, site kind and test
WORKERS = 4  # processes recomputing the cases
# A scene line starts with {"scene_id": "<id>", so the id is read without parsing the JSON.
SCENE_ID_START = 14  # len('{"scene_id": "')
SCENE_ID_END = 64  # upper bound on the id's column (scene ids are up to 50 characters)


def render_cases(
    labels: Sequence[str | Path],
    scenes: Sequence[str | Path],
    out: str | Path,
    spec: str,
    *,
    closed: Sequence[str | Path] = (),
    per_kind: int = CASES_PER_KIND,
    workers: int = WORKERS,
    run_json: str | None = None,
) -> Path:
    """Pick cases from ``oncue label`` outputs, replay them and write the page; return its path."""
    out = Path(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    picked = pick_cases([Path(p) for p in labels], per_kind, [Path(p) for p in closed])
    found = _load_scenes(scenes, {c["scene_id"] for c in picked})
    tasks = [
        (found[c["scene_id"]], c["sample"], spec, run_json)
        for c in picked
        if c["scene_id"] in found
    ]
    index = []
    for payload in run_jobs(_case_payload, tasks, workers):
        i = len(index)
        write_case(out, i, payload)
        index.append(index_row(i, payload))
    write_index(out, index)
    return write_page(out)


def write_case(out: Path, i: int, payload: dict) -> None:
    """Write case ``i``'s payload as the small JS the page loads when the case is selected."""
    (out / "data" / f"case_{i}.js").write_text(
        f"window.CASES=window.CASES||{{}};window.CASES[{i}]={json.dumps(payload)};"
    )


def index_row(i: int, payload: dict) -> dict:
    """Return case ``i``'s entry in the page's case list -- what the list shows and filters on."""
    label, outcome = payload["label"], payload["outcome"]
    return {
        "i": i,
        "scene_id": payload["scene_id"],
        "target": payload["kind"],  # target type
        "kind": label["site_kind"],
        "test": label["test"],
        "W": label["W"],
        "speed": label["speed"],
        "hits": {name: bool(run.get("hit")) for name, run in outcome.items()},
    }


def write_index(out: Path, index: list[dict]) -> None:
    """Write the case index: sorted by target type, site kind, then anticipation tests, then W."""
    index.sort(key=lambda r: (r["target"], r["kind"], r["test"] != "anticipation", r["W"]))
    (out / "data" / "index.json").write_text(json.dumps(index))


def pick_cases(labels: list[Path], per_kind: int, closed: list[Path] = ()) -> list[dict]:
    """Return up to ``per_kind`` cases per target type, site kind and test, most telling first.

    A sample with two closed-loop records under its key counts as having none.
    """
    loops = _closed_loop_records(closed)
    return _most_telling(_case_pool(labels, loops), per_kind)


def _closed_loop_records(closed: list[Path]) -> dict:
    """Return each sample's closed-loop record keyed by ``_key``; None when the key is ambiguous."""
    loops = {}
    for path in closed:
        for entry in read_lines(path):
            for row in entry.get("rows", []):
                if "closed_loop" in row:
                    key = _key(entry["scene_id"], row)
                    loops[key] = None if key in loops else row["closed_loop"]
    return loops


def _key(scene_id: str, row: dict) -> str:
    """Return a sample's ``sample_id``; older outputs use the request fields, which can collide."""
    if row.get("sample_id"):
        return row["sample_id"]
    return json.dumps([scene_id, *(row.get(k) for k in KEY_FIELDS)])


def _hit(runs: dict, name: str) -> bool:
    """Return whether the closed-loop run ``name`` hit the target (False when it is missing)."""
    return bool(runs.get(name, {}).get("hit"))


def _case_pool(labels: list[Path], loops: dict) -> dict[tuple, list]:
    """Group reaction and anticipation samples by ``(kind, site_kind, test)``.

    Entries are ``(telling, W, scene_id, row)``. A sample is telling when the masked IDM-base
    planner hits the target and neither its unoccluded run nor masked occlusion-aware planning
    does.
    """
    pool: dict[tuple, list] = {}
    for path in labels:
        for entry in read_lines(path):
            for row in entry.get("rows", []):
                test = row.get("test")
                if "_states" not in row or test not in ("reaction", "anticipation"):
                    continue
                runs = loops.get(_key(entry["scene_id"], row)) or {}
                telling = (
                    _hit(runs, "reactive_masked")
                    and not _hit(runs, "reactive_see")
                    and not _hit(runs, "occlusion_aware_masked")
                )
                group = (row.get("kind"), row.get("site_kind"), test)
                pool.setdefault(group, []).append((telling, row["W"], entry["scene_id"], row))
    return pool


def _most_telling(pool: dict[tuple, list], per_kind: int) -> list[dict]:
    """Return up to ``per_kind`` cases per group, one per scene, telling ones first.

    Within a group, anticipation tests are then ordered by longest W, reaction tests by shortest.
    """
    picked = []
    for (_, _, test), entries in sorted(pool.items(), key=lambda kv: str(kv[0])):
        entries.sort(key=lambda e: (not e[0], -e[1] if test == "anticipation" else e[1]))
        seen = set()
        for _telling, _, scene_id, row in entries:
            if scene_id in seen:
                continue  # at most one case per scene and group
            seen.add(scene_id)
            picked.append({"scene_id": scene_id, "sample": row})
            if len(seen) >= per_kind:
                break
    return picked


def _load_scenes(paths: Sequence[str | Path], wanted: set) -> dict:
    """Return ``{scene_id: scene_dict}`` for the wanted scenes across shards (or directories)."""
    found = {}
    for f in (f for f in scene_inputs(paths) if f.name.endswith(".jsonl.gz")):
        with gzip.open(f, "rt") as fh:
            for line in fh:
                sid = line[SCENE_ID_START:SCENE_ID_END].split('"', 1)[0]
                if sid in wanted:
                    found[sid] = json.loads(line)
        if len(found) == len(wanted):
            return found
    return found


def _case_payload(task: tuple) -> dict:
    from ..closedloop.scene import RunOptions, closed_loop_scene
    from ..core.io import load_config
    from ..generation.context import scene_context
    from ..generation.targets import SPECS

    scene_dict, sample, spec_name, run_json = task
    spec = SPECS[spec_name]
    config = load_config(run_json)
    scene = fill_gaps(scene_from_dict(scene_dict), config.replay)
    ctx = scene_context(scene, config, DriverConfig())
    states = np.asarray(sample["_states"], float)
    valid = np.asarray(sample["_valid"], bool)
    target = Track(spec.track_id, spec.agent_type, spec.size, states, valid, spec.height)
    reference = ctx.reference
    with_target = replace(reference, tracks={**reference.tracks, spec.track_id: target})
    vis = config.visibility
    _, values = target_visibility(with_target, spec.track_id, vis, ctx.static)
    (record,) = closed_loop_scene(
        scene,
        [sample],
        spec,
        config,
        DriverConfig(),
        RunOptions(tuple(CASE_PLANNERS), keep_states=True),
    )
    anchor = scene.anchor
    # On frames without the target, show it at the first position where it does appear.
    shown = np.flatnonzero(valid)
    idx = shown[np.clip(np.searchsorted(shown, np.arange(len(states))), 0, len(shown) - 1)]
    focus = np.vstack([states[valid, :2], reference.ego.states[reference.ego.valid, :2]])
    near = near_box(focus, CASE_VIEW_MARGIN_M)
    tracks = tracks_payload(scene, near)
    egos, outcome = _runs_payload(record["closed_loop"], reference.ego, states, valid, idx)
    label = _label_payload(sample, reference.ego, anchor, scene.dt, ctx.limit)
    xyh = states[idx, :3]
    return {
        "scene_id": scene.scene_id,
        "dt": scene.dt,
        "current_idx": anchor,
        "times": rounded(scene.timestamps - scene.timestamps[anchor]),
        "ego_size": rounded(scene.ego.size),
        "occluder": sample.get("site"),
        "map": map_payload(scene, near),
        "static": static_payload(ctx.static, focus, CASE_VIEW_MARGIN_M),  # walls to draw
        "tracks": tracks,
        "kind": spec.kind,
        "target": {
            "size": rounded(spec.size),
            "xyh": [None if np.isnan(p[0]) else p for p in rounded(xyh)],
        },
        "visible": [None if not np.isfinite(v) else round(float(v), 2) for v in values],
        "threshold": vis.threshold,
        "egos": egos,
        "outcome": outcome,
        "label": label,
    }


def _runs_payload(
    runs: dict,
    reference_ego: Track,
    states: np.ndarray,
    valid: np.ndarray,
    idx: np.ndarray,
) -> tuple[dict, dict]:
    """Return the reference and per-run ego paths, and each run's record without its states.

    ``idx`` maps each frame to the frame the target is shown at.
    """
    egos = {
        "reference": {
            "xyh": rounded(reference_ego.states[:, :3]),
            "v": rounded(reference_ego.states[:, 3]),
        }
    }
    outcome = {}
    for name, result in runs.items():
        if "error" in result:
            outcome[name] = {"error": result["error"]}
            continue
        run_states = np.asarray(result["states"])
        egos[name] = {
            "xyh": rounded(run_states[:, :3]),
            "v": rounded(run_states[:, 3]),
            "until": int(result["valid_until"]),
        }
        # Show the target as this run moved it: a triggered start shifts it with the tested ego.
        moved_target = np.asarray(result["target_states"], float)
        if not np.allclose(moved_target[valid], states[valid], equal_nan=True):
            egos[name]["target"] = [
                None if np.isnan(q[0]) else q for q in rounded(moved_target[idx, :3])
            ]
        outcome[name] = {
            k: v for k, v in result.items() if k not in ("states", "valid_until", "target_states")
        }
    return egos, outcome


def _label_payload(
    sample: dict, reference_ego: Track, anchor: int, dt: float, speed_limit: float
) -> dict:
    """Return the case's label panel data."""
    reveal_frame = anchor + frame_of(sample["t_c"] - sample["W"], dt)
    return {
        "site_kind": sample.get("site_kind"),
        "speed": sample.get("peak_speed"),  # target's peak speed (m/s)
        "contact_key": sample.get("contact_key"),
        **{k: sample.get(k) for k in ("t_r", "t_c", "W")},
        # Share of background-safe candidates that are safe (descriptive).
        "ma": _share(sample.get("oracle_revealed")),
        "ma0": _share(sample.get("oracle_early")),
        "test": sample.get("test"),
        "ego_speed_at_reveal": round(float(reference_ego.states[reveal_frame, 3]), 2),
        "speed_limit": round(speed_limit, 2),
    }


def _share(judged: dict | None) -> float | None:
    """Return safe / background-safe for a condition's judgement, or None when either is zero."""
    if not judged or not judged.get("background_safe"):
        return None
    return round(judged["safe"] / judged["background_safe"], 3)


def write_page(out: str | Path) -> Path:
    """Rebuild a case viewer's page from its stored index without recomputing cases."""
    out = Path(out)
    index = json.loads((out / "data" / "index.json").read_text())
    (out / "index.html").write_text(page_html("cases.html", index))
    return out / "index.html"


def page_html(template: str, index: list) -> str:
    """Return the page template with the site's shared CSS/JS and the case index inlined."""
    t = files("oncue.render").joinpath("templates")
    return (
        t.joinpath(template)
        .read_text()
        .replace("/*COMMON_CSS*/", t.joinpath("common.css").read_text())
        .replace("/*COMMON_JS*/", t.joinpath("common.js").read_text())
        .replace("/*INDEX*/[]", json.dumps(index))
    )
