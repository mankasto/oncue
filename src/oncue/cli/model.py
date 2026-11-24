"""``oncue model``: train, extract data for, recalibrate and validate the trajectory generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .options import given

EXCLUDE_NONE = "none"  # --exclude-scenes value meaning "keep every scene"
RECALIBRATION_BATCHES = 200  # batches BatchNorm statistics are averaged over
RECALIBRATION_SHARDS = 16  # random training shards sampled for recalibration
VALIDATION_SAMPLES = 4096  # the training's validation sample set
VALIDATION_BATCH = 128
COMPARISON_SAMPLES = 4000  # held-out samples the models are compared on


def add_parsers(subcommands: argparse._SubParsersAction) -> None:
    model = subcommands.add_parser("model", help="the trajectory generator (needs PyTorch)")
    steps = model.add_subparsers(dest="model_command", required=True)
    add_extract(steps)
    add_recalibrate(steps)
    add_validate(steps)
    add_compare(steps)
    add_train(steps)


def add_extract(steps: argparse._SubParsersAction) -> None:
    parser = steps.add_parser("extract", help="training samples from raw shards (resumable)")
    parser.add_argument("shards", nargs="+", type=Path, help="raw WOMD Scenario TFRecord shards")
    parser.add_argument("--out", type=Path, required=True, help="feature directory")
    parser.add_argument(
        "--workers",
        type=int,
        help="processes at once, each holding a raw shard (default: models.datasets.waymo."
        "data_preprocess.EXTRACT_WORKERS)",
    )
    parser.add_argument(
        "--records-per-shard", type=int, help="only the first N records of each shard"
    )
    parser.add_argument("--dataset-version", help="WOMD release of the shards (default v1.2.0)")
    parser.add_argument(
        "--split",
        default="hash",
        choices=("hash", "train", "val"),
        help="hash: 20%% of scenes held out; train / val: the whole shard",
    )
    parser.add_argument(
        "--exclude-scenes",
        help="scene ids kept out (one per line; default "
        "oncue/models/datasets/waymo/safeshift_scored.txt, the scenes SafeShift scored, which "
        "SEAL's hard set is drawn from); 'none' keeps every scene",
    )
    parser.set_defaults(handler=run_extract)


def run_extract(args: argparse.Namespace) -> None:
    from ..models.datasets.waymo.data_preprocess import EXCLUDE_DEFAULT, FeatureConfig, extract

    if args.exclude_scenes == EXCLUDE_NONE:
        exclude = None
    else:
        exclude = args.exclude_scenes or EXCLUDE_DEFAULT
    summary = extract(
        args.shards,
        args.out,
        FeatureConfig(),
        limit=args.records_per_shard,
        version=args.dataset_version,
        split=args.split,
        exclude=exclude,
        **given(workers=args.workers),
    )
    shown = {key: value for key, value in summary.items() if key != "results"}
    print(json.dumps(shown, indent=2))


def add_recalibrate(steps: argparse._SubParsersAction) -> None:
    parser = steps.add_parser("recalibrate", help="precise BatchNorm statistics for a checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path, help="feature directory (its training shards)")
    parser.add_argument("--out", type=Path, required=True, help="the recalibrated checkpoint")
    parser.add_argument("--batches", type=int, default=RECALIBRATION_BATCHES)
    parser.add_argument(
        "--shards",
        type=int,
        default=RECALIBRATION_SHARDS,
        help="random training shards to draw from",
    )
    parser.add_argument(
        "--clean-inputs",
        action="store_true",
        help="without the input dropouts (history, route, signals), as validation and generation",
    )
    parser.set_defaults(handler=run_recalibrate)


def run_recalibrate(args: argparse.Namespace) -> None:
    from ..models.eval_utils import recalibrate_checkpoint

    result = recalibrate_checkpoint(
        args.checkpoint, args.data, args.out, args.batches, args.shards, args.clean_inputs
    )
    print(json.dumps(result))


def add_validate(steps: argparse._SubParsersAction) -> None:
    parser = steps.add_parser("validate", help="validation loss and metrics of a checkpoint")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("data", type=Path, help="feature directory (its *.val.npz files)")
    parser.add_argument(
        "--samples",
        type=int,
        default=VALIDATION_SAMPLES,
        help="the training's validation sample set (default %(default)s)",
    )
    parser.add_argument("--batch", type=int, default=VALIDATION_BATCH)
    parser.add_argument(
        "--bn-batch-stats",
        action="store_true",
        help="BatchNorm on each batch's statistics (as in training) instead of the running ones",
    )
    parser.set_defaults(handler=run_validate)


def run_validate(args: argparse.Namespace) -> None:
    from ..models.eval_utils import validate_checkpoint

    result = validate_checkpoint(
        args.checkpoint, args.data, args.samples, args.batch, args.bn_batch_stats
    )
    print(json.dumps(result))


def add_compare(steps: argparse._SubParsersAction) -> None:
    parser = steps.add_parser("compare", help="generator checkpoints on held-out raw scenes")
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--shard", type=Path, required=True, help="held-out raw WOMD shard")
    parser.add_argument("--samples", type=int, default=COMPARISON_SAMPLES)
    parser.add_argument("--dataset-version", help="WOMD release of the shard (default v1.2.0)")
    parser.set_defaults(handler=run_compare)


def run_compare(args: argparse.Namespace) -> None:
    from ..models.compare import CompareConfig, compare

    config = CompareConfig(samples=args.samples)
    result = compare(args.checkpoints, args.shard, config, args.dataset_version)
    print(json.dumps(result, indent=2))


def add_train(steps: argparse._SubParsersAction) -> None:
    parser = steps.add_parser("train", help="train the trajectory generator as in the paper")
    parser.add_argument("data", type=Path, help="feature directory from `oncue model extract`")
    parser.add_argument(
        "--out", type=Path, required=True, help="checkpoint file (written every epoch)"
    )
    parser.add_argument("--epochs", type=int, help="default 30 (resume: the checkpoint's)")
    parser.add_argument("--batch", type=int, help="per process (GPU); default 128")
    parser.add_argument("--lr", type=float, help="learning rate (default 1e-4)")
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, help="bfloat16 autocast on the GPU"
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        help="torch.compile the model (Triton kernels)",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "max-autotune-no-cudagraphs"),
        help="torch.compile mode (CUDA-graph modes are not offered: untested with DDP)",
    )
    parser.add_argument("--weight-decay", type=float, help="AdamW weight decay (paper: 0.01)")
    parser.add_argument("--grad-clip", type=float, help="gradient norm clip (paper: 1000)")
    parser.add_argument(
        "--schedule",
        choices=("step", "onecycle"),
        help="learning rate: step (MTR's, the paper's) or onecycle (warm-up, cosine down)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="stop after N epochs without a better validation loss (0: never); the best "
        "checkpoint is kept as OUT.best.pt",
    )
    parser.add_argument("--resume", action="store_true", help="continue from --out")
    parser.add_argument(
        "--reseed",
        action="store_true",
        help="--resume with another number of processes: weights and optimizer restored, the "
        "random streams new from the seed",
    )
    parser.add_argument(
        "--dense-mask",
        action=argparse.BooleanOptionalAction,
        help="the dense future branch's BatchNorm on valid road users only (default for new "
        "models); --no-dense-mask: every slot, as earlier models",
    )
    parser.add_argument(
        "--start-velocity",
        action=argparse.BooleanOptionalAction,
        help="the trajectory spline starts at the given start velocity (default off, "
        "as earlier models)",
    )
    parser.add_argument(
        "--init",
        type=Path,
        help="start a new run from this checkpoint's weights (fine-tuning: its configuration, "
        "intention points and shards; new optimizer and schedule)",
    )
    parser.add_argument("--shards-in-memory", type=int, help="feature shards loaded at a time")
    parser.add_argument("--limit-shards", type=int, help="only the first N feature shards")
    parser.set_defaults(handler=run_train)


def run_train(args: argparse.Namespace) -> None:
    from dataclasses import replace

    from ..models.config import RouteTrainConfig
    from ..models.train import train_route
    from ..models.train_utils import checkpoint_config

    # options given here override the checkpoint's configuration or the defaults
    if args.resume:
        if not args.out.exists():
            raise SystemExit(f"--resume: no checkpoint at {args.out.resolve()}")
        config = checkpoint_config(args.out)
    elif args.init:  # fine-tuning: start from the init checkpoint's configuration
        config = checkpoint_config(args.init)
    else:
        config = RouteTrainConfig()
    run_options = {
        "bf16": args.bf16,
        "compile": args.compile,
        "compile_mode": args.compile_mode,
        "patience": args.patience,
    }
    config = replace(config, **given(**run_options))
    train_options = {
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
        "shards_in_memory": args.shards_in_memory,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "schedule": args.schedule,
    }
    train = replace(config.train, **given(**train_options))
    if args.dense_mask is not None:
        train = replace(train, model=replace(train.model, dense_mask=args.dense_mask))
    if args.start_velocity is not None:
        train = replace(train, model=replace(train.model, start_velocity=args.start_velocity))
    summary = train_route(
        args.data,
        args.out,
        replace(config, train=train),
        resume=args.resume,
        reseed=args.reseed,
        train_files=args.limit_shards or 0,
        init=args.init,
    )
    print(json.dumps(summary))
