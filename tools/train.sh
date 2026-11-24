#!/bin/bash
# Train the trajectory generator with torchrun: one process per GPU per node, in the current Python
# environment and this checkout.
#   bash tools/train.sh FEATURES OUT.pt [oncue model train options]
# e.g. bash tools/train.sh features runs/model.pt --epochs 2 --bf16 --compile
#
# Single node: torchrun --standalone. Multi-node: every node runs this script with WORLD_SIZE
# (the number of nodes), RANK (this node's index), MASTER_ADDR and MASTER_PORT set (as a
# PyTorchJob launcher sets them); torchrun joins them.
#
# Writes OUT.pt (the latest checkpoint), OUT.best.pt (the best validation), OUT.epochs/ (each
# epoch's weights), OUT.log.jsonl (losses, data wait, validation, timings), OUT.tb (TensorBoard)
# and OUT.gpu.csv (GPU utilization and memory every 10 s; with
# several nodes each gets OUT.gpu<node>.csv).
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"  # make this checkout importable
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}  # per process; several processes share the cores
features=${1:?usage: bash tools/train.sh FEATURES OUT.pt [oncue model train options]}
out=${2:?usage: bash tools/train.sh FEATURES OUT.pt [oncue model train options]}
shift 2
gpus=$(nvidia-smi -L | wc -l)
nodes=${WORLD_SIZE:-1}
node=${RANK:-0}
if [ "$nodes" -gt 1 ]; then
  launch=(--nnodes "$nodes" --node_rank "$node" --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT")
  gpu_log="$out.gpu$node.csv"
else
  launch=(--standalone)
  gpu_log="$out.gpu.csv"
fi
mkdir -p "$(dirname "$out")"
python -c "import torch, shapely, google_crc32c, oncue; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'shapely', shapely.__version__, 'oncue', oncue.__file__)"
echo "node $node of $nodes, $gpus GPUs: torchrun ${launch[*]}"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 10 \
  > "$gpu_log" &
monitor=$!
trap 'kill $monitor 2>/dev/null || true' EXIT
# torchrun hands each process its own WORLD_SIZE (total processes) and RANK (global).
python -m torch.distributed.run "${launch[@]}" --nproc_per_node "$gpus" \
  -m oncue model train "$features" --out "$out" "$@"
