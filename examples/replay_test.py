"""Replay one OnCue test against one planner, as the paper's closed loop does: in the recorded log,
once with masked observation (the planner knows the target only once the ego can see it) and once
with unoccluded observation (it knows the target from the start), then print what each run
counts as.

    python examples/replay_test.py LABELS SCENES... [--sample SAMPLE_ID] [--planner reactive]

LABELS: an `oncue label` output;
SCENES: the scene shards it was generated on (`oncue export`). The planners are
`oncue.closedloop.PLANNERS`; the paper's five are reactive (IDM-base), conservative,
occlusion_aware, pdm_closed and latentdriver (the last needs PyTorch and its weights).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from oncue.closedloop import MODES, PLANNERS, TESTS, RunOptions, closed_loop_scene, run_name
from oncue.core import Config, DriverConfig, Scene, iter_scenes, scene_inputs
from oncue.evaluate import output_rows, run_flags
from oncue.generation import SPECS, usable

MODE_NAMES = {"masked": "masked", "see": "unoccluded"}
# what a run counts as (evaluate.outputs.run_flags), in the order printed
SHOWN_FLAGS = ("hit", "clean", "stalled", "other", "out_of_road", "wrong_way", "beyond_limits")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("labels", type=Path, help="an `oncue label` output")
    parser.add_argument("scenes", nargs="+", type=Path, help="its scene shards")
    parser.add_argument("--sample", help="the test to replay (default: the first)")
    parser.add_argument("--planner", default="reactive", choices=sorted(PLANNERS))
    args = parser.parse_args()

    scene_id, test = pick_test(args.labels, args.sample)
    config = Config()  # the protocol's settings, as the paper's runs used them
    scene = find_scene(args.scenes, scene_id, config)
    spec = SPECS[test["kind"]]
    options = RunOptions(planners=(args.planner,))
    record = closed_loop_scene(scene, [test], spec, config, DriverConfig(), options)
    runs = record[0]["closed_loop"]

    print(f"test {test['sample_id']}: {test['kind']}, W = {test['W']:.1f} s, label {test['test']}")
    for mode in MODES:
        run = runs[run_name(args.planner, mode)]
        flags = run_flags(run)
        shown = ", ".join(f"{name} {'yes' if flags[name] else 'no'}" for name in SHOWN_FLAGS)
        print(f"  {args.planner}, {MODE_NAMES[mode]:10s}: {shown}")
        if run["hit"]:
            print(
                f"    hits the target {run['contact_s']:.1f} s after the start, "
                f"the ego at {run['contact_speed']:.1f} m/s"
            )


def pick_test(labels: Path, sample_id: str | None) -> tuple[str, dict]:
    """(log, row) of the test with `sample_id`, or of the first test the closed loop can run."""
    for scene_id, row in output_rows([labels]):
        runnable = "_states" in row and usable(row) and row.get("test") in TESTS
        if runnable and sample_id in (None, row["sample_id"]):
            return scene_id, row
    raise SystemExit(f"no runnable test {sample_id or ''} in {labels}")


def find_scene(scene_files: list[Path], scene_id: str, config: Config) -> Scene:
    """The log `scene_id` from the scene shards, gap-filled as every test runs on it."""

    def is_wanted(found: str) -> bool:
        return found == scene_id

    for path in scene_inputs(scene_files):
        for scene in iter_scenes(path, replay=config.replay, wanted=is_wanted):
            return scene
    raise SystemExit(f"log {scene_id} is not in the scene files")


if __name__ == "__main__":
    main()
