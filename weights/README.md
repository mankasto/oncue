# Model weights

OnCue reads checkpoints from `weights/<file>`. None is distributed with the code; every output's
`.settings.json` records the md5 of the model it ran.

| File | What it is | Where to get it | md5 |
| --- | --- | --- | --- |
| `trajectory_generator.pt` | The trajectory generator: the hidden target's trajectory from its start, route and timed waypoints. | Not released with the code; train it yourself (below). | yours |
| `latentdriver_j3.ckpt` | LatentDriver with 3 decoder layers (their `lantentdriver_t2_J3.ckpt`). Planner `latentdriver`. | [LatentDriver's release](https://huggingface.co/Sephirex-x/LatentDriver/tree/main/checkpoints) | `690e4e446c9fc11b963f286e36cccdfc` |
| `plant.ckpt` | PlanT retrained in Waymax (their `planT.ckpt`). Planner `plant`. | [LatentDriver's release](https://huggingface.co/Sephirex-x/LatentDriver/tree/main/checkpoints) | `29de5d1a24921ea0a6ea43b1dcbb962f` |

IDM and PDM-Closed need no weights. Rule-based generation (`oncue generate --generators rule`)
needs no model.

## Training the trajectory generator

On the 1,000 WOMD v1.3.1 training shards, 30 epochs and then 4 epochs from the best one with the
start velocity at learning rate 2e-5 (the checkpoint records these settings):

```bash
python -m oncue model extract $WOMD/training.tfrecord-* --out features/
bash tools/train.sh features runs/route.pt --bf16
bash tools/train.sh features runs/model.pt --init runs/route.best.pt --start-velocity \
  --lr 2e-5 --epochs 4 --bf16
cp runs/model.best.pt weights/trajectory_generator.pt
```

`tools/train.sh` runs one process per GPU with torchrun, on one node or several.
