"""`oncue label`: run the oracle against generated samples and add the labels to the generated
lines (`label`, per scene `label_scene`)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..core.config import DriverConfig, with_emergency_jerk
from ..core.io import iter_scenes, jsonable, load_config, read_lines, scene_inputs
from ..core.runs import (
    PARTS,
    SceneWriter,
    error_fields,
    resume_done,
    run_parts,
    settings_path,
    write_or_check,
)
from ..core.types import Track
from ..core.visibility import find_reveals, target_visibility
from ..oracle.labels import label_sample
from ..utils.parallel import run_jobs
from .checks.validity import contact_frame, reveal_before_contact
from .context import scene_context
from .settings import check_upstream, run_settings
from .targets import SPECS

if TYPE_CHECKING:  # import only for type hints
    from collections.abc import Iterable
    from typing import Any

    from ..core.config import Config
    from ..core.record_types import TestRow
    from ..core.types import Scene
    from .context import SceneContext
    from .targets import TargetSpec

# fields a label step writes into a sample row, removed before (re)labelling
LABEL_FIELDS = (
    "test",
    "judge_missing",
    "W0",
    "budget",
    "candidate_list",
    "reveal_frame",
    "early_visible_frame",
    "evaluation_end_frame",
    "core_coverage_complete",
    "oracle_revealed",
    "oracle_early",
    "braking_suffices",
    "error",
    "tb",
    "hidden_by",
    # fields of earlier label files
    "ma",
    "ma0",
    "test_strict",
    "strict_missing",
    "delta_act_s",
    "dense_ma",
    "emergency_escape",
    "random_escape",
)
_LINES: dict = {}  # scene id -> generated line to label; set once per worker process


def label(
    generated: str | Path,
    scenes: Iterable[str | Path],
    out_path: str | Path,
    spec: str,
    workers: int = 1,
    parts: int = PARTS,
    run_json: str | Path | None = None,
    budget: int | None = None,
    brake_jerk: float | None = None,
    relabel: tuple[str, ...] | None = None,
) -> int:
    """Label every sample of an `oncue generate` output; return the number of lines written.

    Args:
        budget: overrides the oracle's candidates per condition.
        brake_jerk: overrides the emergency brake's jerk, in m/s^3.
        relabel: `generated` is a label output; only samples with these labels are relabelled
            and the rest are left out.
    """
    config = _config(run_json, budget, brake_jerk)
    check_upstream(generated, spec, SPECS[spec], config)
    text = Path(generated).read_text()
    lines = {d["scene_id"]: d for d in read_lines(generated)}
    if relabel is not None:
        lines = {k: kept for k, d in lines.items() if (kept := _relabelled(d, relabel))}
    if any(r.get("found") and "_states" not in r for d in lines.values() for r in d["rows"]):
        raise SystemExit(f"{generated}: samples without their states (an older generate output)")
    files = scene_inputs(scenes)
    out_path = Path(out_path)
    # generators come from the settings: a run with no realizable request has no row naming them
    generate_settings = json.loads(settings_path(generated).read_text())
    settings = run_settings(
        "label",
        spec,
        SPECS[spec],
        config,
        files,
        generated=str(generated),
        generated_sha256=hashlib.sha256(text.encode()).hexdigest(),
        generators=generate_settings.get("generators"),
        run_json=run_json,
    )
    write_or_check(out_path, settings)
    skip = resume_done(out_path)
    todo = {k: d for k, d in lines.items() if k not in skip}
    with out_path.open("a") as fh:
        writer = SceneWriter(fh, skip)
        # nothing to label: the generate line passes through unchanged
        writer.write([json.dumps(d) for d in todo.values() if not _has_sample(d)])
        pending = {k: d for k, d in todo.items() if _has_sample(d)}
        opts = {"spec": spec, "run_json": run_json, "budget": budget, "brake_jerk": brake_jerk}
        items = [(*part, opts) for part in run_parts(files, parts)]
        # each worker gets the lines once, not with every job
        for out in run_jobs(_job, items, workers, initializer=_init, initargs=(pending,)):
            writer.write(out)
    writer.report(out_path)
    missing = sorted(set(pending) - writer.ids)
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(pending)} scenes with samples not in the scene inputs, "
            f"e.g. {missing[0]}; {writer.written} written"
        )
    return writer.written


def _config(run_json: str | Path | None, budget: int | None, brake_jerk: float | None = None):
    """Return the run's config with `budget` and `brake_jerk` overriding the file's."""
    config = load_config(run_json)
    if budget is not None:
        config = replace(config, oracle=replace(config.oracle, budget=budget))
    if brake_jerk is not None:
        candidates = with_emergency_jerk(config.oracle.candidates, brake_jerk)
        config = replace(config, oracle=replace(config.oracle, candidates=candidates))
    return config


def _relabelled(line: dict, labels: tuple[str, ...]) -> dict | None:
    """Return the line keeping only samples whose current label is in `labels`; None if none."""
    rows = [r for r in line["rows"] if _is_sample(r) and r.get("test") in labels]
    return {**line, "rows": rows} if rows else None


def _has_sample(line: dict) -> bool:
    return any(_is_sample(r) for r in line.get("rows", []))


def _init(lines: dict) -> None:
    global _LINES
    _LINES = lines


def _job(item: tuple):
    path, part, parts, opts = item
    spec = SPECS[opts["spec"]]
    config = _config(opts.get("run_json"), opts.get("budget"), opts.get("brake_jerk"))
    out = []
    for scene in iter_scenes(path, part, parts, config.replay, wanted=_LINES.__contains__):
        line = _LINES[scene.scene_id]
        try:
            rows = label_scene(scene, line["rows"], spec, config, DriverConfig())
        except Exception as e:  # any other scene failure: samples stay unlabelled, with the error
            rows = _with_error(line["rows"], e)
        out.append(json.dumps(jsonable({**line, "rows": rows})))
    return out


def label_scene(
    scene: Scene, rows: list[dict], spec: TargetSpec, config: Config, driver: DriverConfig
) -> list[dict]:
    """Return the scene's rows with its samples labelled, or carrying the error instead."""
    rows = [_unlabelled(r) if _is_sample(r) else r for r in rows]
    try:
        ctx = scene_context(scene, config, driver)
    except Exception as e:  # scene-level failure
        return _with_error(rows, e)
    out = []
    for row in rows:
        if _is_sample(row):
            try:
                row = {**row, **label_row(ctx, spec, row)}
            except Exception as e:  # no label at all: keep only the error
                row = {**row, **error_fields(e)}
        out.append(row)
    return out


def _is_sample(row: Any) -> bool:
    return isinstance(row, dict) and bool(row.get("found")) and "_states" in row


def _unlabelled(row: TestRow) -> TestRow:
    """Return the sample row without the fields an earlier labelling added."""
    return {k: v for k, v in row.items() if k not in LABEL_FIELDS}


def _with_error(rows: list[TestRow], e: BaseException) -> list[TestRow]:
    """Return the scene's rows with every sample carrying the error instead of a label."""
    err = error_fields(e)
    return [{**_unlabelled(r), **err} if _is_sample(r) else r for r in rows]


def label_row(ctx: SceneContext, spec: TargetSpec, row: TestRow) -> TestRow:
    """Return the label fields for one generated sample row on its scene context."""
    anchor, dt, vis = ctx.anchor, ctx.dt, ctx.config.visibility
    states = np.asarray(row["_states"], float)
    valid = np.asarray(row["_valid"], bool)
    f_r, f_c = int(row["f_reveal"]), int(row["f_contact"])
    target = Track(spec.track_id, spec.agent_type, spec.size, states, valid, spec.height)
    ref = ctx.reference
    f = contact_frame(ctx, spec, states, valid)
    if f != f_c:
        raise ValueError(f"the contact at frame {f_c} is found at frame {f}")
    with_target = replace(ref, tracks={**ref.tracks, spec.track_id: target})
    hits, values = target_visibility(with_target, spec.track_id, vis, ctx.static)
    event = next((e for e in find_reveals(values, vis, dt) if anchor < e.reveal == f_r), None)
    if event is None:
        raise ValueError(f"the reveal at frame {f_r} is not found again")
    labels = label_sample(
        with_target,
        spec.track_id,
        event,
        f_c,
        vis=vis,
        static=ctx.static,
        oracle=ctx.oracle,
        hits=hits,
        cutoff=row.get("cutoff_frame"),
    )
    return {**labels, "hidden_by": hidden_by(ctx, with_target, spec.track_id, f_r, f_c)}


def hidden_by(
    ctx: SceneContext, with_target: Scene, target_id: str, f_reveal: int, f_contact: int
) -> str:
    """Return "road_users" when road users alone hide the sample until `f_reveal`, else
    "walls"."""
    if ctx.static is None:
        return "road_users"
    vis = ctx.config.visibility
    _, values = target_visibility(with_target, target_id, vis, None)
    why, event = reveal_before_contact(values, vis, ctx.dt, ctx.anchor, f_contact)
    return "road_users" if why is None and event.reveal == f_reveal else "walls"


def usable(row: Any) -> bool:
    """Return whether a labelled sample is a usable test with no error."""
    return _is_sample(row) and "test" in row and "error" not in row
