# closedloop

Runs labelled tests against the planners under test (`oncue closedloop`). Each test sample runs
against each planner in two observation modes: masked (the planner knows only what the ego can
see) and unoccluded. A standing target starts when the tested ego approaches (the triggered
start); a moving target keeps its fixed timing; background road users replay the log and never
react. Start from `closed_loop_scene` in `scene.py`.

## Commands

```
oncue closedloop LABELLED SCENES... --out OUT.jsonl --spec SPEC
```

- `--planners`: names in `registry.PLANNERS`; default `registry.DEFAULT_PLANNERS` (listed in
  [planners/](../planners/README.md)). Other registered planners, such as
  `latentdriver_finetuned`, run only when named.
- `--modes masked,see`, `--fixed` (fixed timing for standing targets too), `--tests` (default
  reaction, anticipation, unresolved; `all` for every usable sample).
- `--world replay|metadrive` (default `replay`). `metadrive` gives the same runs, mirrored in
  MetaDrive only to draw them.
- It resumes scene by scene.

The case pages (`oncue cases`, `oncue site`) reuse the parts in `scene.py`. An outside planner
joins with `--planner-module` (see the main README and `examples/my_planner.py`).

## Files

| File | What it holds |
| --- | --- |
| `run.py` | `oncue closedloop`: reads the label output, writes the records |
| `scene.py` | one scene's runs: every planner, test row and mode |
| `records.py` | what one run records: contacts, braking, motion, road use |
| `registry.py` | the planner table (`PLANNERS`, `DEFAULT_PLANNERS`, `register`) |
| `execution.py` | moves every planner's ego with `core.vehicle.execute` |
| `trigger.py` | starts a standing target as the tested ego approaches |
| `metadrive_world.py` | mirrors the replay world in MetaDrive for drawing |

The drive loop (`drive`, `ReplayWorld`) is in [planners/driver.py](../planners/README.md); the
reference ego in [generation/reference.py](../generation/README.md).

## Inputs and outputs

- In: an `oncue label` output and its scene shards. Only usable samples with `_states` run.
- Out: JSONL, one line per scene: `{"scene_id", "rows"}`, one record per test row. A record's
  `closed_loop` holds one run per planner and mode, keyed `<planner>_<mode>` (for example
  `pdm_closed_masked`), with the contact with the target (`hit`), background contacts judged by
  nuPlan's at-fault rules, the ego's peak measures and its road use.
- A run that raised keeps its error. A scene that failed as a whole gives every row failed runs,
  so the rates keep them in the denominator.
- `<output>.settings.json` records the planners, world, modes, tests and the label output's
  sha256.

## Dependencies

Imports `utils/`, `core/`, `data/`, `oracle/`, `generation/` and `planners/`; from `generation/` the
tests it runs and the pipeline's settings record it checks its input against (`run_settings`,
`check_upstream`). `planners/` never
imports `closedloop/`: the closed loop sets up the runs and records them, while the drive loop
and the planners live in `planners/`.
