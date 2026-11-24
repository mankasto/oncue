"""The trajectory generator's configurations, MTR's `mtr/config.py` with its yaml settings: the
model's (`ModelConfig`), its training's (`TrainConfig`) and the training run's (`RouteTrainConfig`:
the training, input dropouts, precision and compilation, early stop, kept epochs, precise
BatchNorm, validation, seed). Each has one entry that reads a checkpoint's stored one:
`model_config`, `train_config`, and for `RouteTrainConfig` `train_utils.checkpoint_config`."""

from __future__ import annotations

from dataclasses import dataclass, field, fields


@dataclass(frozen=True)
class ModelConfig:
    """Defaults: the paper model, MTR scale (MTR's context encoder 256 wide with feed-forward
    1,024, map segments by its PointNet, local attention to the 16 nearest inputs; agent encoding,
    dense future prediction and motion decoder 512 wide; 6 + 6 layers, 8 heads, 64 intention
    points per type)."""

    d_model: int = 512  # the decoder's width
    intention_points: int = 64
    encoder_layers: int = 6
    decoder_layers: int = 6
    heads: int = 8
    future: int = 80
    encoder_width: int = 256  # the context encoder's width (MTR: 256)
    encoder_ff: int = 4  # the encoder's feed-forward width per channel
    local_neighbours: int = 16  # encoder attention to the k nearest inputs (MTR: 16)
    # the only local attention built: the k neighbours' keys and values gathered, as MTR's
    local_attention: str = "gather"
    # the dense future branch's BatchNorm counts the valid road users only; False: every slot,
    # padding included (models trained before 10-23)
    dense_mask: bool = True
    # the trajectory spline's first derivative at the origin is the start velocity, given (the
    # center's current velocity, where known) or else predicted, instead of left free by a
    # natural spline (models before 10-25: False)
    start_velocity: bool = False

    def __post_init__(self):
        if self.encoder_width < 1:
            raise ValueError("encoder_width: the context encoder is at least 1 wide")
        if self.local_neighbours < 1:
            raise ValueError("local_neighbours: the encoder attends to at least 1 input")
        if self.local_attention != "gather":
            raise ValueError(f"local_attention {self.local_attention!r}: only gather")


# A configuration stored before 10-27 carries the fields of the removed model (rs_w352's decoder
# and compact inputs): accepted only with the value this model was always built with (absent in
# such a record: that model's default, refused); the fields only that model read are ignored.
_ONLY = {"inputs": ("compact", "mtr"), "decoder": ("simple", "mtr"),
         "map_encoder": ("mlp", "pointnet")}  # fmt: skip
_UNUSED = ("time_encoding", "agent_size", "map_direction")
_ADDED = {"dense_mask": False, "start_velocity": False}  # missing in older ones: their behaviour


def model_config(stored: dict) -> ModelConfig:
    """The ModelConfig of a checkpoint's stored one (`train.model`), the one entry for loading,
    resuming and fine-tuning; a configuration this code cannot build is refused (ValueError)."""
    stored = dict(stored)
    before = any(k in stored for k in (*_ONLY, *_UNUSED))  # stored before 10-27
    for key, (old_default, only) in _ONLY.items():
        value = stored.pop(key, old_default if before else only)
        if value != only:
            raise ValueError(f"model {key} {value!r}: only {only!r} (rs_w352's model is removed)")
    for key in _UNUSED:
        stored.pop(key, None)
    missing = [
        item.name
        for item in fields(ModelConfig)
        if item.name not in stored and item.name not in _ADDED
    ]
    if missing:
        raise ValueError(f"model configuration without {missing}")
    return ModelConfig(**{**_ADDED, **stored})


@dataclass(frozen=True)
class TrainConfig:
    """Defaults: the paper model's training, MTR's optimization (AdamW, learning rate 1e-4 held,
    then halved every 2 epochs from 22/30 of the epochs; weight decay 0.01, gradient norm clip
    1,000), 30 epochs, batch 128 per process, 4 feature shards in memory."""

    epochs: int = 30
    batch: int = 128
    lr: float = 1e-4
    weight_decay: float = 0.01
    # gradient norm clip; learning-rate schedule "step" (MTR's: lr held, then times lr_decay
    # every 2 epochs from 22/30 of the epochs, not below lr_min) or "onecycle" (warm-up to lr,
    # cosine down)
    grad_clip: float = 1000.0
    schedule: str = "step"
    lr_decay: float = 0.5
    lr_min: float = 1e-6
    waypoint_counts: tuple[float, ...] = (0.1, 0.2, 0.5, 0.2)  # P(0, 1, 2, 3 waypoints)
    contact_frames: tuple[int, int] = (14, 80)
    window_frames: tuple[int, int] = (5, 51)
    noise_m: float = 0.3
    history_dropout: float = 0.1
    start_weight: float = 5.0
    slot_weights: tuple[float, ...] = (1.0, 3.0, 1.0)  # contact, reveal, extra
    log_every: int = 50  # steps between TensorBoard loss points
    shards_in_memory: int = 4  # 0: load every feature shard; else stream groups of this many
    max_bad_batches: int = 200  # consecutive non-finite losses before training stops
    model: ModelConfig = field(default_factory=ModelConfig)


_TRAIN_UNUSED = ("eval_every", "eval_samples")  # read by the removed training loop only


def train_config(stored: dict) -> TrainConfig:
    """The TrainConfig of a checkpoint's stored one (its `route_train.train`), its model through
    `model_config`."""
    stored = {key: value for key, value in stored.items() if key not in _TRAIN_UNUSED}
    return TrainConfig(**{**stored, "model": model_config(stored["model"])})


@dataclass(frozen=True)
class RouteTrainConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    p_route_off: float = 0.5  # route not given
    p_sig_off: float = 0.2  # signal phases not given
    p_state_only: float = 0.1  # past dropped, the t0 state (pose and speed) kept
    bf16: bool = False  # bfloat16 autocast on the GPU (faster; numerics differ from fp32)
    compile: bool = False  # torch.compile the model (Triton kernels; the first steps compile)
    compile_mode: str = "default"  # or max-autotune-no-cudagraphs (Triton picks matmul configs)
    patience: int = 0  # stop after this many epochs without a better validation loss (0: never)
    # every epoch's weights kept apart (<out>.epochs/epochNN.pt, no optimizer state), and the
    # whole checkpoint of the last epoch before the step schedule's first decay (<out>.pre_decay.pt,
    # to resume from with more epochs, or to start from with --init under another schedule)
    keep_epochs: bool = True
    # precise BatchNorm: before each validation and save, the running statistics
    # recomputed as the plain average over this many batches drawn across bn_shards random
    # training shards (0: off); the momentum-0.1 statistics remember only the last shard group
    bn_batches: int = 200
    bn_shards: int = 16
    val_files: int = 10  # validation shards for the per-epoch validation loss (0: none)
    val_samples: int = 4096
    seed: int = 0
