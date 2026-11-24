# planners

The planners under test: the IDM planner in three settings, PDM-Closed and LatentDriver. A planner
only decides where the ego goes next: `step(frame, ego_states, scene, beliefs)` returns the ego's
next state (x, y, heading, speed). `driver.py` runs a planner through a replayed scene; the executor
and the target's motion are in `closedloop/`. Start from `perception.py` and `idm/planner.py`.

## Names

Names for `oncue closedloop --planners` (`closedloop.registry.PLANNERS`). These eight run by
default (`closedloop.registry.DEFAULT_PLANNERS`):

| Name | Planner |
| --- | --- |
| `reactive`, `conservative`, `occlusion_aware` | the IDM planner with the options in `closedloop.registry.DRIVER_OPTIONS` |
| `pdm_closed` | PDM-Closed |
| `pdm_closed_side_fault` | PDM-Closed counting side impacts as at fault (supplement) |
| `latentdriver` | LatentDriver |
| `latentdriver_native` | LatentDriver moved by its own displacement, not executed (comparison) |
| `latentdriver_log_route` | LatentDriver given the logged ego's whole trajectory as its route (comparison) |

Run only when named: `latentdriver_finetuned` and `latentdriver_finetuned_logs` (LatentDriver fine-tuned on OnCue's tests, and its logs-only control), and `plant`, `plant_finetuned`, `plant_finetuned_logs` (PlanT from LatentDriver's release, run in the same planner, and the same two fine-tuned versions).

## Files

| File | What it holds |
| --- | --- |
| `perception.py` | what a planner knows: unoccluded or masked (`Perception`, `Beliefs`) |
| `driver.py` | the drive loop (`drive`, `ReplayWorld`): a planner through a replayed scene, frame by frame |
| `step_records.py` | per-step records for planners that move the ego themselves |
| `idm/planner.py` | `IDMPlanner`, a rule-based driver along the logged path; the reference ego uses it too |
| `idm/crossings.py` | crossing lanes and phantom vehicles for the occlusion-aware setting |
| `idm/path.py` | the path the IDM planner drives, and shared helpers |
| `pdm/` | PDM-Closed ported from tuplan_garage `b51d5d04` (Apache-2.0, `LICENSE` there): `planner.py`, `proposals.py`, `scorer.py` (nuPlan's tracker and bicycle come from `core/tracker.py`) |
| `latentdriver/model.py` | builds and loads LatentDriver's model, and PlanT's (`load_plant`) |
| `latentdriver/observation.py` | LatentDriver's Waymax observation, rebuilt from an OnCue scene |
| `latentdriver/planner.py` | `LatentDriverPlanner` |
| `latentdriver/sort_vertices.py` | a PyTorch port of LatentDriver's CUDA vertex sort |
| `latentdriver/bert-mini/config.json` | the BERT config its encoder reads |
| `latentdriver/original/` | LatentDriver's model code and method config at commit `721bd6e`, with the same release's PlanT, the original code kept as released (byte-identical, `provenance.json` with every file's sha256, `LICENSE`); never edited, adapted in the files above |

## Inputs and outputs

- In, per run: the scene (with the target inserted), a `core.config.DriverConfig`, and each frame
  the ego's states so far and the perception's `Beliefs`.
- Out, per step: the ego's next state. `closedloop/` writes the run records.
- Weights: LatentDriver reads `weights/latentdriver_j3.ckpt`, PlanT `weights/plant.ckpt`; the IDM and PDM-Closed need none.
  See [weights/README.md](../../../weights/README.md).

## Dependencies

Imports only `utils/`, `core/` and `data/`. It never imports `closedloop/` or `generation/`: the
closed loop and the reference ego use the planners and `driver.py`, never the other way round.
