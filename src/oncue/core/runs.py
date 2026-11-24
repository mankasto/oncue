"""Run bookkeeping every stage shares: settings files, resuming, parts of a shard, and the
scene writer."""

from __future__ import annotations

import hashlib
import json
import os
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from .io import file_sha256

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable, Iterable
    from typing import Any, TextIO

# settings fields that may differ between parts of one run, ignored when comparing
VOLATILE = ("inputs", "input_sha256", "created")
TRACEBACK_CHARS = 800  # a failed row or scene keeps the end of its traceback
PARTS = 8  # jobs per scene shard (`run_parts`) unless a run asks for another number


def input_digests(inputs: Iterable[str | Path]) -> dict[str, str]:
    """Return {absolute input path: sha256 of its content}."""
    return {os.path.abspath(path): file_sha256(path) for path in inputs}


def model_info(path: str | Path | None) -> dict | None:
    if path is None:
        return None
    data = Path(path).read_bytes()
    return {"path": str(path), "md5": hashlib.md5(data).hexdigest(), "bytes": len(data)}


def error_fields(error: BaseException) -> dict:
    """Return what a failed row or scene records: the error and the end of its traceback."""
    trace = "".join(traceback.format_exception(error))
    return {"error": repr(error), "tb": trace[-TRACEBACK_CHARS:]}


def settings_path(out_path: str | Path) -> Path:
    """Return where an output's settings record is: `<out>.settings.json`."""
    return Path(out_path).with_suffix(".settings.json")


def scene_rows(line: dict) -> list:
    """Return the rows of one output line (some outputs call them samples)."""
    return line.get("rows") or line.get("samples") or []


def line_has_error(line: dict) -> bool:
    """Return whether an output line or a row in it carries an error; a resume does it again."""
    return "error" in line or any(_has_error(row) for row in scene_rows(line))


def resume_done(out_path: str | Path) -> set[str]:
    """Return the scenes already done in an output; lines with an error are removed so a resumed
    run does them again."""
    out_path = Path(out_path)
    if not out_path.exists():
        return set()
    done, dropped = set(), set()
    tail, tail_number = None, None  # a cut-short last line and its number
    ended = True  # the file ends with a newline
    with out_path.open() as file:
        line = file.readline()
        n = 0
        while line:
            following = file.readline()
            n += 1
            ended = line.endswith("\n")
            text = line.rstrip("\n")
            if text:
                try:
                    d = json.loads(text)
                except json.JSONDecodeError as e:
                    if not following:  # last line, probably a cut-short write; redo its scene
                        tail, tail_number = text, n
                        line = following
                        continue
                    raise SystemExit(
                        f"{out_path}: line {n} is not JSON; a damaged output, not resumed"
                    ) from e
                if line_has_error(d):
                    dropped.add(n)
                else:
                    done.add(d["scene_id"])
            line = following
    if tail is not None:  # moved aside for inspection, out of the output
        aside = out_path.with_name(out_path.name + ".cut")
        aside.write_text(tail + "\n")
        print(f"{out_path}: an incomplete last line moved to {aside.name}, its scene done again")
    if dropped or tail is not None or not ended:
        # every line complete and newline-terminated before appending
        _rewrite_lines(out_path, lambda number: number not in dropped and number != tail_number)
        if dropped:
            print(f"{out_path}: {len(dropped)} scenes with an error taken out, to be done again")
    return done


def done_in_both(path: Path, done: set[str], other: Path, other_done: set[str]) -> set[str]:
    """Return the scenes both outputs have done; a scene done in only one is taken out of it."""
    for out, scenes in ((path, done - other_done), (other, other_done - done)):
        if scenes:
            drop_scenes(out, scenes)
            print(f"{out}: {len(scenes)} scenes the other output lacks taken out, to be done again")
    return done & other_done


def drop_scenes(path: Path, scenes: set[str]) -> None:
    """Take the lines of `scenes` out of an output, line by line."""
    with path.open() as file:
        numbers = {
            number
            for number, line in enumerate(file, 1)
            if line.strip() and json.loads(line)["scene_id"] in scenes
        }
    _rewrite_lines(path, lambda number: number not in numbers)


def _rewrite_lines(path: Path, keep: Callable[[int], bool]) -> None:
    """Rewrite `path` atomically, keeping the non-empty lines whose 1-based number `keep`
    accepts."""
    tmp = path.with_name(path.name + ".tmp")
    with path.open() as source, tmp.open("w") as target:
        for number, line in enumerate(source, 1):
            text = line.rstrip("\n")
            if text and keep(number):
                target.write(text + "\n")
    os.replace(tmp, path)


def _has_error(item: Any) -> bool:
    """Return whether an output row, a closed-loop run in it or a nominal run carries an
    error."""
    if not isinstance(item, dict):
        return False
    runs = (item.get("closed_loop") or {}).values()
    return (
        "error" in item
        or bool(item.get("nominal_errors"))
        or any(isinstance(r, dict) and "error" in r for r in runs)
    )


def write_or_check(out_path: str | Path, settings: dict) -> Path:
    """Write `<out>.settings.json`; on a resumed run, exit unless the settings and the inputs'
    content match."""
    path = settings_path(out_path)
    out = Path(out_path)
    if not path.exists() and out.exists() and out.stat().st_size > 0:
        raise SystemExit(
            f"{out}: results but no {path.name}; write to a new output instead of resuming"
        )
    if path.exists():
        old = json.loads(path.read_text())
        keys = (set(old) | set(settings)) - set(VOLATILE)
        diff = sorted(k for k in keys if old.get(k) != settings.get(k))
        if diff:
            raise SystemExit(
                f"{path}: settings differ from the earlier run in {diff}; "
                "write to a new output instead of resuming"
            )
        changed = changed_inputs(old, settings)
        if changed:
            raise SystemExit(
                f"{path}: the content of {changed} changed since the earlier run; write to a new "
                "output instead of resuming"
            )
        recorded = with_new_inputs(old, settings)
        if recorded != old:
            path.write_text(json.dumps(recorded, indent=1) + "\n")
        return path
    path.write_text(json.dumps(settings, indent=1) + "\n")
    return path


def changed_inputs(old: dict, new: dict) -> list[str]:
    """Return the input paths in both records whose content's sha256 differs."""
    before, now = old.get("input_sha256") or {}, new.get("input_sha256") or {}
    return sorted(path for path in before.keys() & now.keys() if before[path] != now[path])


def with_new_inputs(old: dict, new: dict) -> dict:
    """Return the record with the inputs a resumed run reads for the first time added."""
    digests, inputs = old.get("input_sha256") or {}, old.get("inputs") or []
    added = {
        path: sha for path, sha in (new.get("input_sha256") or {}).items() if path not in digests
    }
    more = [path for path in new.get("inputs") or [] if path not in inputs]
    if not added and not more:
        return old
    return {**old, "inputs": [*inputs, *more], "input_sha256": {**digests, **added}}


def run_parts(files: Iterable[Path], parts: int) -> list[tuple[str, int, int]]:
    """Return (input, part, its parts) for every part of the inputs, in input order."""
    if not (isinstance(parts, int) and parts >= 1):
        raise ValueError(f"parts {parts!r}: jobs per shard, a whole number >= 1")
    jobs = []
    for f in files:
        count = scene_parts(f, parts)
        jobs += [(str(f), part, count) for part in range(count)]
    return jobs


def scene_parts(path: Path, parts: int) -> int:
    """Return how many parts an input is cut into: `parts` for a shard, one for a scene file."""
    return parts if Path(path).name.endswith(".jsonl.gz") else 1


class SceneWriter:
    """Write output lines once per scene; a scene already written is counted in `duplicates`."""

    def __init__(self, fh: TextIO, already: Iterable[str] = ()):
        self.fh = fh
        self.ids = set(already)
        self.written = 0
        self.duplicates = 0

    def write(self, lines: Iterable[str]) -> None:
        for line in lines:
            sid = json.loads(line)["scene_id"]
            if sid in self.ids:
                self.duplicates += 1
                continue
            self.fh.write(line + "\n")
            self.ids.add(sid)
            self.written += 1
        self.fh.flush()

    def report(self, out_path: str | Path) -> None:
        """Report duplicates and fail the job when any output row carries an error."""
        if self.duplicates:
            print(
                f"{out_path}: {self.duplicates} lines of scenes already written skipped "
                "(a scene in more than one input)"
            )
        failed = error_scenes(out_path)
        if failed:
            raise SystemExit(f"{out_path}: {failed} scenes with an error")


def error_scenes(path: str | Path) -> int:
    """Return how many output lines carry an error (`line_has_error`)."""
    failed = 0
    for line in Path(path).read_text().splitlines():
        if '"error' not in line:  # most lines: no need to parse ("error", "nominal_errors")
            continue
        failed += line_has_error(json.loads(line))
    return failed
