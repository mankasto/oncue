"""Browse OnCue tests: how many a label output holds by target type, label and requested reaction
window W, and the timing of one of them.

    python examples/browse_tests.py LABELS... [--sample SAMPLE_ID]

LABELS: `oncue label` outputs (JSON lines, one per scene).
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from oncue.evaluate import output_rows

# the paper's names of the labels an `oncue label` output gives
LABEL_NAMES = {
    "reaction": "Reactive",
    "anticipation": "Anticipatory",
    "unresolved": "unresolved",
    "not_judged": "not judged",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", nargs="+", type=Path, help="`oncue label` outputs")
    parser.add_argument("--sample", help="the test whose timing is shown (default: the first)")
    args = parser.parse_args()

    tests = [(scene_id, row) for scene_id, row in output_rows(args.labels) if "test" in row]
    if not tests:
        raise SystemExit("no labelled tests in the inputs")
    print_counts(tests)
    print_timing(*pick(tests, args.sample))


def print_counts(tests: list[tuple[str, dict]]) -> None:
    """Tests by target type and label, and by requested W."""
    by_label = Counter((row["kind"], row["test"]) for _, row in tests)
    print(f"{len(tests)} tests in {len({scene_id for scene_id, _ in tests})} logs")
    for (target_type, label), count in sorted(by_label.items()):
        print(f"  {target_type:10s} {LABEL_NAMES[label]:12s} {count:6d}")
    by_window = Counter(row["W_req"] for _, row in tests)
    print("by requested W (s): " + ", ".join(f"{w}: {n}" for w, n in sorted(by_window.items())))


def pick(tests: list[tuple[str, dict]], sample_id: str | None) -> tuple[str, dict]:
    """The test with `sample_id`, or the first one."""
    if sample_id is None:
        return tests[0]
    for scene_id, row in tests:
        if row["sample_id"] == sample_id:
            return scene_id, row
    raise SystemExit(f"no test {sample_id} in the inputs")


def print_timing(scene_id: str, row: dict) -> None:
    """One test's facts. Times are seconds after the test starts: the target is hidden from the
    reference ego until its exposure (t_r) and meets it at contact (t_c); W is the time between."""
    print(f"\ntest {row['sample_id']} in log {scene_id}")
    print(
        f"  {row['kind']} at a {row['site_kind']} site, hidden by {row['hidden_by']}, "
        f"label {LABEL_NAMES[row['test']]}"
    )
    print(
        f"  exposure at {row['t_r']:.1f} s, contact at {row['t_c']:.1f} s: "
        f"W = {row['W']:.1f} s (requested {row['W_req']:.1f} s)"
    )


if __name__ == "__main__":
    main()
