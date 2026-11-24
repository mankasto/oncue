"""Build and load LatentDriver's model, or PlanT from the same release (``original/``, unchanged).

The BERT config the original fetches from the hub is read from ``bert-mini/`` instead.
"""

from __future__ import annotations

import contextlib
import io
import sys
import types
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import yaml

if TYPE_CHECKING:
    from typing import NoReturn

from . import PLANT_WEIGHTS, WEIGHTS

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "original"
BERT_CONFIG_DIR = HERE / "bert-mini"
NUM_OF_DECODER = 3  # J3, LatentDriver's main-table checkpoint
# PlanT's constructor defaults to two bicycle outputs; planT.yaml's action space and the released
# checkpoint have three waypoint outputs (dx, dy, dyaw).
PLANT_CONTROL_TYPE = "waypoint"


def load_model(weights: str | Path = WEIGHTS) -> torch.nn.Module:
    """Return the J3 model with ``weights`` loaded strictly, in eval mode, on the CPU."""
    _original_on_path()
    from src.policy.latentdriver.lantentdriver_model import LantentDriver

    return _loaded(LantentDriver, method_config(), weights)


def load_plant(weights: str | Path = PLANT_WEIGHTS) -> torch.nn.Module:
    """Return PlanT with ``weights`` loaded strictly, in eval mode, on the CPU."""
    _original_on_path()
    from src.policy.baseline.bc_baseline import Simple_driver

    return _loaded(Simple_driver, plant_config(), weights, control_type=PLANT_CONTROL_TYPE)


# Model name -> (its loader, the released checkpoint), for the fine-tuning entry point.
RELEASES = {"latentdriver": (load_model, WEIGHTS), "plant": (load_plant, PLANT_WEIGHTS)}


def _loaded(model_class: type, config: _Config, weights: str | Path, **options) -> torch.nn.Module:
    """Build ``model_class`` from ``config`` and load ``weights`` into it strictly."""
    with contextlib.redirect_stdout(io.StringIO()):  # the original prints while building
        model = model_class(**config, **options)
    state = torch.load(Path(weights), map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.eval()


def _original_on_path() -> None:
    """Put ``original/`` (package ``src``) on the path, stubbing the CUDA op of its IoU loss."""
    if "src.utils.iou" not in sys.modules:
        utils = types.ModuleType("src.utils")
        utils.__path__ = [str(ORIGINAL / "src" / "utils")]  # its other modules import as usual
        iou = types.ModuleType("src.utils.iou")

        def cal_iou(*args, **kwargs) -> NoReturn:
            raise NotImplementedError("LatentDriver's IoU loss (a CUDA op) is for training only")

        iou.cal_iou = cal_iou
        utils.iou = iou
        sys.modules["src.utils"], sys.modules["src.utils.iou"] = utils, iou
    if str(ORIGINAL) not in sys.path:
        sys.path.insert(0, str(ORIGINAL))


def method_config() -> _Config:
    """Compose the method config as hydra does for ``method=latentdriver`` with three decoders."""
    method = _composed("latentdriver.yaml")
    method["num_of_decoder"] = NUM_OF_DECODER
    method["max_length"] = method["max_len"]  # ${max_len}
    method["eval_context_length"] = method["max_len"]  # ${max_len}
    return _attributes(method)


def plant_config() -> _Config:
    """Compose the method config as hydra does for ``method=planT``."""
    method = _composed("planT.yaml")
    method["scheduler"]["max_lr"] = method["learning_rate"]  # ${learning_rate}
    return _attributes(method)


def _composed(name: str) -> dict:
    """Read a method config with its default groups, its encoder set up as upstream's train.py."""

    def read(file: str) -> dict:
        return yaml.safe_load((ORIGINAL / "configs" / "method" / file).read_text())

    method = read(name)
    groups = method.pop("defaults")  # e.g. [{"encoder": "bert"}, {"world": "latent_world_model"}]
    for group in groups:
        ((key, option),) = group.items()
        method[key] = read(f"{key}/{option}.yaml")
    method["encoder"]["embedding_type"] = method["action_space"]["dynamic_type"]
    method["encoder"]["type_name"] = str(BERT_CONFIG_DIR)
    return method


class _Config(dict):
    """Dict that also supports attribute access, mirroring the original's OmegaConf reads."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _attributes(value: object):
    """Recursively wrap dicts in ``_Config`` so they support attribute access."""
    if isinstance(value, dict):
        return _Config({key: _attributes(item) for key, item in value.items()})
    return value
