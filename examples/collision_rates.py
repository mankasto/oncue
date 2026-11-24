"""Target collision rates of the closed loop by target type, label and planner, with masked and
with unoccluded observation, each with its 95 % interval from 10,000 bootstrap resamples (by ego
drive with --drives: logs of the same drive are resampled together; else by log).

The rates are over the tests in the outputs, each test counted once. The paper's planner tables
also weight every stratum (target type, site kind, label, W) to all generated tests; the
per-test rates here are their unweighted columns.

    python examples/collision_rates.py CLOSEDLOOP... [--drives DRIVES] [--planners a,b]

CLOSEDLOOP: `oncue closedloop` outputs; DRIVES (optional): a file mapping each log to its
drive, {"drives": {log: drive}}.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from oncue.closedloop import MODES, run_name
from oncue.evaluate import Estimate, output_rows, rate, run_flags, use_drives

# the paper's planners (closedloop.registry names): IDM-base, IDM-conservative,
# IDM-occlusion-aware, PDM-Closed, LatentDriver
PAPER_PLANNERS = "reactive,conservative,occlusion_aware,pdm_closed,latentdriver"
MODE_NAMES = {"masked": "masked", "see": "unoccluded"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("closedloop", nargs="+", type=Path, help="`oncue closedloop` outputs")
    parser.add_argument("--drives", type=Path, help="drives file: resample by drive")
    parser.add_argument("--planners", default=PAPER_PLANNERS, help="comma list")
    args = parser.parse_args()

    use_drives(args.drives)  # before reading: the drive map also leaves out the logs it drops
    tests = defaultdict(list)  # (target type, label) -> [(log, row)]
    for scene_id, row in output_rows(args.closedloop):
        if "closed_loop" in row:
            tests[(row["kind"], row["test"])].append((scene_id, row))

    header = "".join(f"{MODE_NAMES[mode] + ' collision (%)':>26s}" for mode in MODES)
    print(f"{'target':10s} {'label':12s} {'planner':16s} {'tests':>5s}{header}")
    for (target_type, label), rows in sorted(tests.items()):
        for planner in args.planners.split(","):
            ran = [(scene_id, row) for scene_id, row in rows if has_runs(row, planner)]
            if not ran:
                continue
            cells = "".join(f"{shown(collision_rate(ran, planner, mode)):>26s}" for mode in MODES)
            print(f"{target_type:10s} {label:12s} {planner:16s} {len(ran):5d}{cells}")


def has_runs(row: dict, planner: str) -> bool:
    """Whether the test was run with `planner` in every observation mode."""
    return all(run_name(planner, mode) in row["closed_loop"] for mode in MODES)


def collision_rate(rows: list[tuple[str, dict]], planner: str, mode: str) -> Estimate:
    """The share of `rows` in which `planner` hits the target under `mode` (a target contact
    before the test's cutoff, `run_flags`), with its interval."""
    name = run_name(planner, mode)
    return rate(rows, lambda row: run_flags(row["closed_loop"][name])["hit"])


def shown(estimate: Estimate) -> str:
    return f"{estimate.value:5.1f} [{estimate.low:5.1f}, {estimate.high:5.1f}]"


if __name__ == "__main__":
    main()
