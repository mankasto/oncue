# models

The trajectory generator and the code to train and check it. Given a scene, the inserted target's
start, its route, the signal phases and up to three timed waypoints, the model gives a set of
scored candidate trajectories for the target; one model serves vehicles, pedestrians and
cyclists. Generation loads it in `generation/motion/generators.py`. Start from
`modeling/model.py`.

The structure follows MTR's official code ([sshaoshuai/MTR](https://github.com/sshaoshuai/MTR),
commit `a5ba7bd`) file by file, rewritten with OnCue's additions; a file with an MTR counterpart
names it in its docstring (MTR's `mtr/models` is `modeling/` here). None of these files is an
unchanged copy of MTR.

## Commands

| Command | What it does |
| --- | --- |
| `oncue model extract RAW_SHARDS... --out FEATURES` | training features per scene (numpy only); the scenes in `datasets/waymo/safeshift_scored.txt` are kept out by default (`--exclude-scenes`) |
| `oncue model train FEATURES --out CKPT` | trains the model; several GPUs and nodes with `torchrun` through `tools/train.sh` |
| `oncue model recalibrate CKPT FEATURES --out CKPT2` | precise BatchNorm statistics for a checkpoint |
| `oncue model validate CKPT FEATURES` | validation loss and metrics of a checkpoint |
| `oncue model compare CKPT... --shard RAW_SHARD` | checkpoints compared on held-out raw scenes |
| `oncue generate ... --model weights/trajectory_generator.pt` | generation with the model |

Feature extraction needs only numpy; everything else needs PyTorch (the `model` extra in
pyproject.toml). The checkpoint used in the paper is `weights/trajectory_generator.pt`: see
[weights/README.md](../../../weights/README.md).

## Files

| File | What it holds |
| --- | --- |
| `config.py` | the model, training and training-run configurations |
| `datasets/waymo/data_preprocess.py` | `oncue model extract`: features per scene, one file per shard and split |
| `datasets/waymo/waymo_dataset.py` | scene-level inputs shared by training and generation (`FeatureConfig`) |
| `datasets/waymo/safeshift_scored.txt` | scene ids kept out of training by default |
| `datasets/routes.py` | the route input as polyline chunks (numpy only) |
| `datasets/dataset.py` | the training data: feature shards, background loading, batches |
| `modeling/` | the model: `model.py` (`RouteModel`, `load_route_model`), `context_encoder/`, `motion_decoder/`, `utils/` |
| `ops/` | row gather and k nearest neighbours, in place of MTR's CUDA ops |
| `utils/` | losses, the output's spline bases, intention points |
| `train.py`, `train_utils.py` | `oncue model train`: the loop, processes, checkpoints, learning rate |
| `eval_utils.py` | `oncue model validate` and `recalibrate`, and per-epoch validation |
| `compare.py` | `oncue model compare` |
| `inputs.py` | one inserted track's model inputs, as generation and `compare` call the model |

## Inputs and outputs

- `extract`: raw WOMD Scenario TFRecords in; a feature directory out, with `<shard>.<split>.npz`
  per shard, `features.json` (the feature settings), `excluded.json` (the scenes kept out) and
  `extract.jsonl` (per-shard counts). It resumes.
- `train`: the feature directory in; a checkpoint written every epoch, with the optimizer state,
  so `--resume` continues it.
- In generation: a checkpoint in (`load_route_model`); per start, the candidates in score order.

## Dependencies

Imports only `utils/`, `core/` and `data/`. Only `generation/` and the command line import it.
