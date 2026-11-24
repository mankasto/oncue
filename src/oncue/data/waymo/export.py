"""``oncue export``: WOMD TFRecord shards to scene files (gzipped JSONL), resumable per shard.

Usage: ``oncue export RAW_SHARDS... --out DIR --dataset-version v1.3.1 [--split SPLIT]``
"""

from __future__ import annotations

import gzip
import io
import json
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.io import SCENE_SHARD_SUFFIX, file_sha256, jsonable, same_name_groups, write_json
from ...utils.parallel import run_jobs
from .scenes import load_record
from .split import partition
from .tfrecord import index_records

if TYPE_CHECKING:
    from collections.abc import Iterable

EXPORT_WORKERS = 4  # concurrent processes (each holds one raw shard in memory)
SCENE_GZIP_LEVEL = 6  # gzip compression level for scene files
# Fix the gzip header mtime so re-exporting the same shard yields the same bytes.
SCENE_GZIP_MTIME = 0


def export(
    shards: Iterable[str | Path],
    out_dir: str | Path,
    workers: int = EXPORT_WORKERS,
    *,
    split: str,
    dataset_version: str,
) -> dict:
    """Export every raw shard of ``shards`` to ``out_dir``, skipping gsutil ``.gstmp`` partials.

    ``split`` is the ``split.partition`` kept (``train``, ``internal_validation``) or ``all``, not
    WOMD's own split. Returns ``{shard name: scenes written, or -1 if already done}``.
    """
    files = [Path(s) for s in shards if not Path(s).name.endswith(".gstmp")]
    clashes = same_name_groups(files)
    if clashes:
        raise SystemExit(f"raw shards with one file name would write one scene file: {clashes}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # One directory holds one export; a resumed run must match the recorded settings.
    settings = {"split": split, "dataset_version": dataset_version}
    record = out_dir / "export.json"
    if record.exists():
        old = json.loads(record.read_text())
        diff = sorted(k for k in set(old) | set(settings) if old.get(k) != settings.get(k))
        if diff:
            raise SystemExit(f"{out_dir}: exported with other settings ({diff}); use a new dir")
    elif list(out_dir.glob(f"*{SCENE_SHARD_SUFFIX}")):
        raise SystemExit(f"{out_dir}: scenes without a record of their export; use a new dir")
    else:  # atomic: shard jobs start together and one may read this while another writes it
        write_json(record, settings, indent=1)
    done = {}
    job = partial(export_shard, out_dir=out_dir, split=split, dataset_version=dataset_version)
    for name, n in run_jobs(job, files, workers):
        done[name] = n
        print(name, "already done" if n < 0 else f"{n} scenes", flush=True)
    return done


def export_shard(
    shard: str | Path,
    out_dir: str | Path,
    *,
    split: str,
    dataset_version: str,
) -> tuple[str, int]:
    """Export one raw shard to ``<shard>.scenes.jsonl.gz``, then mark it done with its sha256.

    Returns ``(shard name, scenes written)``, or ``(shard name, -1)`` if it was already done.
    """
    shard, out_dir = Path(shard), Path(out_dir)
    out = out_dir / f"{shard.name}{SCENE_SHARD_SUFFIX}"
    mark = out_dir / f"{shard.name}.done"
    shard_sha256 = file_sha256(shard)
    if mark.exists():
        if json.loads(mark.read_text())["sha256"] != shard_sha256:
            raise SystemExit(
                f"{shard}: its content changed since {out.name} was exported from it; use a new dir"
            )
        return shard.name, -1
    tmp = out_dir / f".{shard.name}.scenes.jsonl.gz.tmp"
    n = 0
    packed = gzip.GzipFile(tmp, "wb", SCENE_GZIP_LEVEL, mtime=SCENE_GZIP_MTIME)
    with packed, io.TextIOWrapper(packed) as f:
        for ref in index_records(shard, None):
            scene = load_record(ref, dataset_version=dataset_version)
            if split != "all" and partition(scene.scene_id) != split:
                continue
            f.write(json.dumps(jsonable(scene), allow_nan=False) + "\n")
            n += 1
    tmp.rename(out)
    write_json(mark, {"shard": shard.name, "sha256": shard_sha256, "scenes": n}, indent=1)
    return shard.name, n
