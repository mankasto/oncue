# generation

Makes the hidden-target tests (`oncue generate`) and labels them with the oracle (`oncue label`).
For each scene and target type it finds where a hidden target can cross the ego's path, builds
requests (where the target touches the reference ego, and the window W in which it is visible
before that), and keeps the candidate motion that passes every hard check with the fewest flaws.
Start from `pipeline.py`.

## Commands

| Command | What it does |
| --- | --- |
| `oncue generate SCENES... --out OUT.jsonl --spec SPEC --model weights/trajectory_generator.pt` | the tests of one target type |
| `oncue label GENERATED SCENES... --out OUT.jsonl --spec SPEC` | adds the oracle's labels to a generate output |

- `--spec` is a name in `targets.SPECS`: `pedestrian`, `cyclist` or `vehicle`.
- `generate` options: `--generators` (default `model,rule`; `contact_only,free,grid` for the
  ablation), `--raw-candidates N`, `--device` (`cpu` by default, `cuda`, `cuda:N`, `mps`).
- `label` options: `--budget N` sets the oracle budget.
- Both commands resume: scenes already in the output are skipped. A resumed run whose settings,
  code or inputs changed stops.
- `experiments/run_shards.py` runs them as the `generate`, `label` and `ablation` stages.

## Files

| File | What it holds |
| --- | --- |
| `pipeline.py` | `oncue generate`: the per-scene loop over routes, requests and starts |
| `search.py` | picks the passing candidate with the fewest flaws |
| `rows.py` | the rows `generate` writes, and the per-scene funnel |
| `label.py` | `oncue label`: adds the oracle's labels to generated samples |
| `targets.py` | what differs between target types (`TargetSpec`, `SPECS`) |
| `context.py` | per-scene state shared by every step |
| `settings.py` | the settings every stage records about the whole pipeline (`run_settings`), and the check that an input was made under the same ones |
| `reference.py` | the protocol reference ego (driven by `planners/driver.py`) |
| `requests/sites.py` | where a hidden target can cross the ego's path |
| `requests/build.py` | requests on a route: contact point and window W |
| `motion/starts.py` | the target's starts for a request, commonest first |
| `motion/generators.py` | candidate motions from the learned model and the rule profiles |
| `checks/validity.py` | hard checks that reject a candidate (`check_candidate`) |
| `checks/realism.py` | flaws recorded on a passing candidate (`sample_labels`) |
| `checks/scene.py` | per-scene inputs the checks read |
| `checks/diagnostics.py` | the raw-candidate diagnostic (`--raw-candidates`) |

## Inputs and outputs

- In: scene files from `oncue export`, the trajectory generator's checkpoint (see
  [weights/README.md](../../../weights/README.md)) and an optional `--config` JSON.
- `generate` writes JSONL, one line per scene: `{"scene_id", "rows", "funnel"}`, with one row per
  request and generator variant (model or rule). With `--raw-candidates`, the diagnostic goes to
  `<out>.raw_candidates.jsonl`.
- `label` writes the same lines with labels added to each sample row: `test` (reaction,
  anticipation, unresolved or not_judged), `oracle_revealed` and `oracle_early`.
- Every output has a `<output>.settings.json` with its settings and each input's sha256.

## Dependencies

Imports `utils/`, `core/`, `data/`, `oracle/`, `planners/` and `models/` (the last only inside the
functions that run the learned trajectory generator). `settings.py` reads `closedloop/` constants inside a
function: one settings record covers the whole pipeline, so a generation output also records the
closed loop's constants, and a closed-loop run checks its input against the same record. Inside
the package, `motion/` may import `requests/`, never the reverse.
