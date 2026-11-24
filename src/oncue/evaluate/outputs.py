"""Shared rules for reading generate, label and closed-loop outputs and their settings records,
each rule in one place."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..closedloop.run import TESTS as CLOSED_LOOP_TESTS  # the labels run in the closed loop
from ..core.io import read_lines
from ..core.runs import VOLATILE, scene_rows, settings_path
from . import bootstrap

if TYPE_CHECKING:  # annotations only
    from collections.abc import Callable, Iterable, Iterator, Sequence

    from ..core.record_types import TestRow

LABELS = (*CLOSED_LOOP_TESTS, "not_judged")  # every label value

# A setting naming an input file has the file's digest under ``<setting>_sha256``.
DIGEST_SUFFIX = "_sha256"

PER_KIND = ("spec_name", "spec", "driver")  # per-target-type settings


def output_rows(paths: Sequence[str | Path], errors: bool = False) -> Iterator[tuple[str, dict]]:
    """Iterate ``(scene_id, row)`` pairs from the output files.

    Skips rows with an error unless ``errors``, and the logs the active drive map drops.
    """
    for d in scene_lines(paths):
        for x in scene_rows(d):
            if isinstance(x, dict) and (errors or "error" not in x):
                yield d["scene_id"], x


def failures(paths: Sequence[str | Path]) -> dict:
    """Return failure counts by kind; ``run_failed`` runs also count in the rates."""
    out = Counter()
    for _, x in output_rows(paths, errors=True):
        if isinstance(x.get("closed_loop"), dict):
            out.update(run_failed=sum("error" in r for r in x["closed_loop"].values()))
            out.update(nominal_failed=len(x.get("nominal_errors") or {}))
        if "error" not in x:
            continue
        if x.get("found"):
            out["label_failed"] += 1
        elif x.get("kind") is not None:
            out["route_failed"] += 1
        else:
            out["scene_failed"] += 1
    return {k: v for k, v in out.items() if v}


def funnel_table(paths: Sequence[str | Path]) -> dict:
    """Return ``{kind: totals}`` of the scene funnels; ``mismatch`` holds unbalanced scenes."""
    out: dict = {}
    for d in scene_lines(paths):
        f = d.get("funnel")
        if not f:
            continue
        t = out.setdefault(f["kind"], defaultdict(Counter))
        t["scenes"][f["scene"]] += 1
        if f["scene"] != "ok":
            continue
        considered, dropped = Counter(f["considered"]), Counter(f["dropped"])
        t["considered"].update(considered)
        t["dropped"].update(dropped)
        t["routes"]["routes"] += f["routes"]
        t["routes"]["failed"] += f["route_errors"]
        t["found"].update(f["found"])
        lost = Counter()
        for key, n in dropped.items():
            lost[key.split("/")[0]] += n
        kept = {stage: considered[stage] - lost[stage] for stage in considered}
        ok = kept.get("route", 0) == f["routes"]
        ok &= considered.get("request", 0) == kept.get("contact", 0) * f["windows"]
        if not ok:
            t["mismatch"][d["scene_id"]] += 1
    return out


def methods(runs: dict) -> dict:
    """Return ``{(run, generator): [(scene_id, request_row)]}`` for ``{name: output_paths}``.

    A generator's rows are its realizable requests plus the run's unrealizable ones.
    """
    out = {}
    for run, paths in runs.items():
        requests = [
            (scene_id, row)
            for scene_id, row in output_rows(paths, errors=True)
            if "request_id" in row
        ]
        unrealizable = [
            (scene_id, row) for scene_id, row in requests if row.get("generator") is None
        ]
        named = {row.get("generator") for _, row in requests} - {None}
        for generator in sorted(named | asked_generators(paths)):
            rows = [
                (scene_id, row) for scene_id, row in requests if row.get("generator") == generator
            ]
            out[(run, generator)] = rows + unrealizable
    return out


def asked_generators(paths: Iterable[str | Path]) -> set[str]:
    """Return the generators generate was asked for, read from the outputs' settings."""
    asked = set()
    for path in paths:
        if settings_path(path).exists():
            asked |= set(output_settings(path).get("generators") or ())
    return asked


def group_by(items: Iterable, key: Callable) -> dict:
    """Return ``{key(item): [items]}`` sorted by the string representation of the key."""
    out = defaultdict(list)
    for item in items:
        out[key(item)].append(item)
    return dict(sorted(out.items(), key=lambda pair: str(pair[0])))


def fields(row: TestRow, by: Sequence[str]) -> tuple:
    """Return a tuple of ``row[field]`` for each field in ``by``; missing fields become ``None``."""
    return tuple(row.get(field) for field in by)


def is_anticipation(row: TestRow) -> bool:
    """Return whether the row's test label is ``"anticipation"``."""
    return row.get("test") == "anticipation"


def output_settings(path: str | Path) -> dict:
    """Return the ``<output>.settings.json`` for an output."""
    return json.loads(settings_path(path).read_text())


def shared_config(paths: Sequence[str | Path]) -> dict:
    """Return the config the outputs ``paths`` all record; ValueError unless exactly one."""
    configs = {json.dumps(output_settings(path).get("config"), sort_keys=True) for path in paths}
    if len(configs) != 1:
        raise ValueError(f"label outputs made with {len(configs)} configs")
    return json.loads(configs.pop())


def labels_by_sample(paths: Iterable[str | Path]) -> dict:
    """Return ``{sample_id: (scene_id, row)}`` for the samples found in label outputs."""
    return {
        row["sample_id"]: (scene_id, row)
        for scene_id, row in output_rows(paths, errors=True)
        if row.get("found")
    }


ORACLE_RUNS = {"revealed": "oracle_revealed", "early": "oracle_early"}  # a label's two searches


def oracle_runs(row: TestRow) -> dict:
    """Return a labelled row's oracle runs by condition.

    ``"early"`` is left out when it reuses the revealed run.
    """
    runs = {name: row.get(key) for name, key in ORACLE_RUNS.items()}
    early = row.get("early_visible_frame")
    if early is not None and early == row.get("reveal_frame"):
        runs.pop("early")
    return {name: run for name, run in runs.items() if isinstance(run, dict)}


def oracle_work(rows: list[tuple[str, dict]]) -> dict:
    """Return per-condition oracle-work totals: runs, candidates attempted and executed, seconds.

    ``seconds`` is None when a run has none recorded.
    """
    out = {}
    for name in ORACLE_RUNS:
        runs = [run for _, row in rows if (run := oracle_runs(row).get(name))]
        seconds = [run.get("elapsed_s", 0.0) for run in runs]
        out[name] = {
            "runs": len(runs),
            "attempted": sum(run.get("attempted", 0) for run in runs),
            "executed": sum(run.get("executed", 0) for run in runs),
            "seconds": None if None in seconds else round(sum(seconds), 3),
        }
    return out


def output_lines(paths: Iterable[str | Path]) -> list[dict]:
    """Return the scene lines of ``paths`` as a list."""
    return list(scene_lines(paths))


def scene_lines(paths: Iterable[str | Path]) -> Iterator[dict]:
    """Iterate scene lines one at a time, skipping logs the active drive map drops."""
    for path in paths:
        for entry in read_lines(path):
            if not bootstrap.is_dropped(entry["scene_id"]):
                yield entry


def shared_settings(record: Path) -> dict:
    """Return the settings shared by every output of one run: no volatile keys or own inputs."""
    loaded = json.loads(record.read_text())
    settings = without_own_inputs({k: v for k, v in loaded.items() if k not in VOLATILE})
    # ``raw_candidates.file`` names a file beside each output.
    if isinstance(settings.get("raw_candidates"), dict):
        settings["raw_candidates"] = {
            k: v for k, v in settings["raw_candidates"].items() if k != "file"
        }
    return settings


def without_own_inputs(settings: dict) -> dict:
    """Return ``settings`` without each key that has a ``<key>_sha256`` sibling, nor the sibling.

    Nested dicts too; a digest with no matching key stays.
    """
    own = set()
    for key in settings:
        named = key.removesuffix(DIGEST_SUFFIX)
        if named != key and named in settings:
            own |= {named, key}
    return {
        key: without_own_inputs(value) if isinstance(value, dict) else value
        for key, value in settings.items()
        if key not in own
    }


# ---- W-balanced list: rates weighted to all its strata's tests -------------------------------


@dataclass(frozen=True)
class Population:
    """The strata of a test list, loaded from its ``list.json``.

    Attributes:
        path: source file for diagnostics.
        by: stratum field names, in order.
        tests: per-stratum count of eligible tests before the quota.
        chosen: per-stratum count of tests the list actually took.
        stratum_of: ``{sample_id: stratum_key}`` for every listed sample.
    """

    path: str
    by: tuple
    tests: dict
    chosen: dict
    stratum_of: dict

    def key(self, row: TestRow) -> tuple:
        """Return a row's stratum key; raises if its sample is not in the list."""
        if row.get("sample_id") not in self.stratum_of:
            raise ValueError(f"sample {row.get('sample_id')} is not in the list {self.path}")
        return self.stratum_of[row["sample_id"]]

    def weights(self, keys: Sequence[tuple]) -> dict:
        """Return ``{stratum: weight}`` for the rows in each stratum of ``keys``.

        A stratum weighs its total tests times the share of its listed tests among the rows.
        """
        rows = Counter(keys)
        return {key: self.tests[key] * n / self.chosen[key] for key, n in rows.items()}


def read_population(path: str | Path) -> Population:
    """Return the ``Population`` of a list's ``list.json`` and the label outputs beside it."""
    path = Path(path)
    record = json.loads(path.read_text())
    by = tuple(record["settings"]["by"])
    tests, chosen = {}, {}
    for stratum in record["strata"]:
        key = tuple(stratum[field] for field in by)
        tests[key], chosen[key] = stratum["tests"], stratum["chosen"]
    stratum_of = {
        row["sample_id"]: fields(row, by)
        for subset in sorted(path.parent.glob("*.jsonl"))
        for line in read_lines(subset)
        for row in line.get("rows", [])
    }
    if unknown := sorted(set(stratum_of.values()) - tests.keys(), key=str):
        raise ValueError(f"{path}: listed samples in strata {unknown[:3]} it does not count")
    return Population(str(path), by, tests, chosen, stratum_of)


# Resource accounting: what a generator's search spent.
def full_checks(row: TestRow) -> int:
    """Return how many candidates a request's search ran through the full checks."""
    return int(row.get("tried") or 0)
