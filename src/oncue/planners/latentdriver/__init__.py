"""LatentDriver (Xiao et al., ICRA 2025; Sephirex-X/LatentDriver, Apache-2.0) in the closed loop.

``original/`` is the unchanged upstream code; start reading at ``planner.LatentDriverPlanner``.
"""

from pathlib import Path

# Kept here, not in model.py, so they can be used without importing torch.
WEIGHTS_DIR = Path(__file__).resolve().parents[4] / "weights"
WEIGHTS = WEIGHTS_DIR / "latentdriver_j3.ckpt"  # released J3 checkpoint
FINETUNED_WEIGHTS = WEIGHTS_DIR / "latentdriver_j3_finetuned.ckpt"  # OnCue tests + logged driving
FINETUNED_LOGS_WEIGHTS = WEIGHTS_DIR / "latentdriver_j3_finetuned_logs.ckpt"  # logged driving only
# PlanT from the same release (its planT.ckpt), and the same two fine-tuned versions of it.
PLANT_WEIGHTS = WEIGHTS_DIR / "plant.ckpt"
PLANT_FINETUNED_WEIGHTS = WEIGHTS_DIR / "plant_finetuned.ckpt"
PLANT_FINETUNED_LOGS_WEIGHTS = WEIGHTS_DIR / "plant_finetuned_logs.ckpt"
