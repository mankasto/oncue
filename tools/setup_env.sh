#!/bin/bash
# Build OnCue's environment, once, on a Linux x86_64 machine.
# Installs environment.yml's Python and pins and pulls MetaDrive's assets.
#   bash tools/setup_env.sh [ENV]        (from the code folder; about 15 min)
#
# ENV is the environment's folder (default: ONCUE_HOME/envs/oncue); elsewhere, also set
# ONCUE_PY=ENV/bin/python for run_all.sh.
#
# Two ways to build it:
#   default          a fresh Python from conda-forge (PATH's conda, or Miniforge installed here;
#                    uv when conda-forge is unreachable), with CPU torch
#   CLONE_FROM=SRC   a clone of the conda environment SRC (e.g. one with CUDA torch), with
#                    environment.yml's pins installed into the clone; SRC is not changed
#
# Paths come from the environment:
#   ONCUE_HOME  work folder (default this repository); downloads go into ONCUE_HOME/.cache
#   CONDA_ROOT  where Miniforge goes when there is no conda (default ONCUE_HOME/miniforge3)
set -euo pipefail
CODE=$(cd "$(dirname "$0")/.." && pwd)
ONCUE_HOME=${ONCUE_HOME:-$CODE}
ENV=${1:-$ONCUE_HOME/envs/oncue}
CLONE_FROM=${CLONE_FROM:-}
CONDA_ROOT=${CONDA_ROOT:-$ONCUE_HOME/miniforge3}
CACHE_DIR=$ONCUE_HOME/.cache  # Miniforge installer, uv's Python cache
export UV_PYTHON_INSTALL_DIR=$CACHE_DIR/uv-python
TORCH_CPU=https://download.pytorch.org/whl/cpu
PYTHON_VERSION=$(sed -n 's/^ *- python=//p' "$CODE/environment.yml")
mkdir -p "$CACHE_DIR" "$(dirname "$ENV")"
fail() { echo "setup_env: $*" >&2; exit 1; }

echo "== 1. what the build needs"
git --version || fail "MISSING: git (MetaDrive is installed from GitHub)"

echo "== 2. Python $PYTHON_VERSION at $ENV${CLONE_FROM:+ (a clone of $CLONE_FROM)}"
if [ ! -x "$ENV/bin/python" ]; then
  CONDA=$(command -v conda || true)
  if [ -n "$CLONE_FROM" ]; then
    CONDA=${CONDA:-$CLONE_FROM/bin/conda}  # a base env ships conda itself
    "$CONDA" create -y -q -p "$ENV" --clone "$CLONE_FROM"
  else
    if [ -z "$CONDA" ] && [ ! -x "$CONDA_ROOT/bin/conda" ]; then
      curl -fsSL -o "$CACHE_DIR/miniforge.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        && bash "$CACHE_DIR/miniforge.sh" -b -p "$CONDA_ROOT" || echo "no Miniforge"
    fi
    CONDA=${CONDA:-$CONDA_ROOT/bin/conda}
    if ! "$CONDA" create -y -q -p "$ENV" -c conda-forge --override-channels \
        "python=$PYTHON_VERSION" pip; then
      echo "conda-forge failed: Python $PYTHON_VERSION through uv (GitHub) instead"
      rm -rf "$ENV"
      python3 -m pip install -q --user uv
      ~/.local/bin/uv python install "$PYTHON_VERSION"
      ~/.local/bin/uv venv --seed --python "$PYTHON_VERSION" "$ENV"
    fi
  fi
fi
PY=$ENV/bin/python
HAVE_VERSION=$("$PY" -c "import platform; print(platform.python_version())")
echo "Python $HAVE_VERSION"
[ "$HAVE_VERSION" = "$PYTHON_VERSION" ] \
  || echo "NOTE: Python $HAVE_VERSION, not environment.yml's $PYTHON_VERSION: the pins may not install"

TORCH_FROM="CPU wheels"
[ -z "$CLONE_FROM" ] || TORCH_FROM="the clone's, when at the pinned version"
echo "== 3. packages (environment.yml's pip lines; torch, torchvision: $TORCH_FROM)"
REQUIREMENTS=$ENV/oncue-requirements.txt
"$PY" - "$CODE/environment.yml" "$CLONE_FROM" > "$REQUIREMENTS" <<'EOF'
import re, sys
cpu_torch = not sys.argv[2]  # a clone keeps its source's torch (2.8.0+cu128 satisfies ==2.8.0)
for line in open(sys.argv[1]):
    pin = line.split("#")[0].strip()
    if pin.startswith("- ") and ("==" in pin or " @ " in pin):
        pin = pin[2:].strip()
        if cpu_torch:
            pin = re.sub(r"^(torch|torchvision)==(\S+)$", r"\1==\2+cpu", pin)
        print(pin)
EOF
wc -l < "$REQUIREMENTS"
"$PY" -m pip list --format=freeze > "$ENV/pip-freeze-before.txt"
"$PY" -m pip install -q --upgrade "pip<26"
if [ -n "$CLONE_FROM" ]; then
  "$PY" -m pip install -q -r "$REQUIREMENTS"
else
  "$PY" -m pip install -q --extra-index-url "$TORCH_CPU" -r "$REQUIREMENTS"
fi
"$PY" -m pip check || echo "pip check: see above"

echo "== 4. MetaDrive assets"
"$PY" -m metadrive.pull_asset > "$ENV/pull_asset.log" 2>&1 || { tail -5 "$ENV/pull_asset.log"; exit 1; }
tail -1 "$ENV/pull_asset.log"

echo "== 5. installed against the pins, and what the install changed"
"$PY" -m pip list --format=freeze > "$ENV/pip-freeze.txt"
"$PY" - "$REQUIREMENTS" "$ENV/pip-freeze.txt" "$ENV/pip-freeze-before.txt" <<'EOF'
import sys
def pins(path):
    out = {}
    for line in open(path):
        if "==" in line:
            name, version = line.strip().split("==", 1)
            out[name.lower().replace("_", "-")] = version
    return out
wanted, got, before = (pins(path) for path in sys.argv[1:])
def public(version):  # drop any +local label so 2.8.0+cu128 matches torch==2.8.0
    return (version or "").split("+")[0]
wrong = {n: (v, got.get(n)) for n, v in wanted.items() if public(got.get(n)) != public(v)}
print(f"{len(wanted)} pins, {len(wanted) - len(wrong)} as pinned")
for name, (want, have) in sorted(wrong.items()):
    print(f"  NOT AS PINNED: {name} wants {want}, has {have}")
changed = {n: (before[n], got.get(n)) for n in before if got.get(n) != before[n]}
print(f"{len(before)} packages before the install, {len(changed)} of them changed")
for name, (old, new) in sorted(changed.items()):
    print(f"  CHANGED: {name} {old} -> {new}")
EOF

echo "== 6. oncue runs"
PYTHONPATH=$CODE/src "$PY" -c "
import numpy, shapely, tensorboard, tensorflow, torch, oncue
from metadrive.version import VERSION as metadrive_version  # metadrive has no __version__
print('torch', torch.__version__, 'numpy', numpy.__version__, 'shapely', shapely.__version__,
      'tensorflow', tensorflow.__version__, 'tensorboard', tensorboard.__version__,
      'metadrive', metadrive_version)
" 2>&1 | grep -v "^WARNING\|oneDNN\|cuda\|TF-TRT" || true
PYTHONPATH=$CODE/src "$PY" -m oncue --help > /dev/null && echo "oncue: ok"

echo "== 7. the versions OnCue imports, and torch on this machine's GPU"
# Expect CUDA only when we cloned a source environment and the machine has a GPU; else CPU wheels.
CUDA_EXPECTED=""
[ -n "$CLONE_FROM" ] && nvidia-smi -L > /dev/null 2>&1 && CUDA_EXPECTED=yes
PYTHONPATH=$CODE/src "$PY" - "$CUDA_EXPECTED" <<'EOF'
import sys
import numpy, torch, transformers
for module in (numpy, torch, transformers):
    print(f"{module.__name__} {module.__version__}  {module.__file__}")
if not sys.argv[1]:
    print("torch on the CPU (no clone, or no GPU listed by nvidia-smi)")
    sys.exit()
print("torch", torch.__version__, "CUDA", torch.version.cuda, "GPU", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FAILED: a GPU is present but this torch cannot use it")
major, minor = torch.cuda.get_device_capability(0)
gpu_arch = f"sm_{major}{minor}"
print("built for", torch.cuda.get_arch_list(), "this GPU", gpu_arch, torch.cuda.get_device_name(0))
if gpu_arch not in torch.cuda.get_arch_list():
    sys.exit(f"FAILED: this torch is not built for {gpu_arch}")
x = torch.ones(1000, 1000, device="cuda")
print("a product on the GPU:", float((x @ x).sum()))
EOF
echo "== done: $ENV (its packages in $ENV/pip-freeze.txt; run_all.sh: ONCUE_PY=$PY)"
