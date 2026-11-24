# core

Definitions that several packages share: the scene data model, the configuration, file reading
and writing, and the measures and rules the pipeline applies everywhere (kinematics, visibility,
road position, traffic rules, the vehicle model). It has no command of its own. Start from
`types.py` for the data and `config.py` for the parameters.

## Files

| File | What it holds |
| --- | --- |
| `types.py` | the scene data model: `Scene`, `Track`, map features, signals |
| `config.py` | every tunable parameter, grouped by stage (`Config`) |
| `io.py` | JSON and scene files, digests, atomic writes, `load_config`, `iter_scenes` |
| `runs.py` | settings files, resuming, parts of a shard and the scene writer, for every stage |
| `record_types.py` | typed views of the JSON lines the pipeline writes |
| `replay.py` | fills a road user's missing states where the log is unambiguous |
| `visibility.py` | whether the ego can see a target, and reveal events |
| `kinematics.py` | motion measures, physical limits and rare-motion labels |
| `vehicle.py` | executes a desired trajectory within the vehicle limits |
| `tracker.py` | nuPlan's two-stage controller (LQR tracker, kinematic bicycle), ported (Apache 2.0) |
| `fault.py` | nuPlan's at-fault rules for contacts with background road users |
| `cutoff.py` | the frame where a test ends, and why |
| `road.py` | off-road, wrong-way and lane lookups for a box on the map |
| `rules.py` | signals, stop signs and speed limits along the driver's path |

## Using it

- Scene files are `*.scenes.jsonl.gz` shards from `oncue export` (one scene per line) or single
  scene JSON files. `io.load_scene` reads one file; `io.iter_scenes` walks a shard.
- `oncue --config FILE` overrides the defaults with a JSON file; `io.load_config` reads it into
  `Config`, and an unknown key is an error. The values and where they come from are listed in
  [PARAMETERS.md](../../../PARAMETERS.md). Only `Config` fields can be overridden; module
  constants are fixed in the code and recorded in every output's settings.

- Generate, label and closed-loop outputs record the `Config` they ran with in their
  `.settings.json`, so a run resumes only under the same settings.

## Dependencies

Imports only `utils/` and `data/`; every other package imports it. The direction rule: `utils/`
lowest, then `core/` and `data/`, then the rest; a lower package never imports a higher one.
