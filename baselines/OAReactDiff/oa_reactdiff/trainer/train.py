"""Train OA-ReactDiff on any benchmark dataset, supervision level and atom ordering.

Replaces the previous one-script-per-configuration layout; everything that used to
be hardcoded is a command-line argument. See --help for the full list.
"""
import argparse
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

from oa_reactdiff.model import EGNN, LEFTNet
from oa_reactdiff.trainer.ema import EMACallback
from oa_reactdiff.trainer.pl_trainer import DDPMModule

DATASET_DIRS = {
    "ts1x": "transition1x",
    "iso": "isomer",
    "rgd1": "rgd1",
    "rgdx": "rgdx",
    "rgdy": "rgdy",
}
PERMUTATIONS = ["none", "product", "ts", "product+ts"]
MODELS = {"leftnet": LEFTNet, "egnn": EGNN}

EGNN_CONFIG = dict(
    in_node_nf=8,  # embedded dim before injecting to egnn
    in_edge_nf=0,
    hidden_nf=256,
    edge_hidden_nf=64,
    act_fn="swish",
    n_layers=9,
    attention=True,
    out_node_nf=None,
    tanh=True,
    coords_range=15.0,
    norm_constant=1.0,
    inv_sublayers=1,
    sin_embedding=True,
    normalization_factor=1.0,
    aggregation_method="mean",
)
LEFTNET_CONFIG = dict(
    pos_require_grad=False,
    cutoff=10.0,
    num_layers=6,
    hidden_channels=196,
    num_radial=96,
    in_hidden_channels=8,
    reflect_equiv=True,
    legacy=True,
    update=True,
    pos_grad=False,
    single_layer_output=True,
    object_aware=True,
)
MODEL_CONFIGS = {"leftnet": LEFTNET_CONFIG, "egnn": EGNN_CONFIG}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", required=True, choices=sorted(DATASET_DIRS))
    data.add_argument(
        "--fraction",
        type=int,
        default=100,
        choices=[10, 50, 100],
        help="percentage of the training set; selects a prebuilt *_subset_<n>_splits dir",
    )
    data.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="directory holding the <dataset>_splits folders",
    )
    data.add_argument(
        "--datadir",
        type=Path,
        default=None,
        help="explicit split directory; overrides --dataset/--fraction resolution",
    )
    data.add_argument(
        "--permute",
        default="none",
        choices=PERMUTATIONS,
        help="randomly relabel atoms of these fragments in every batch",
    )
    data.add_argument("--num-workers", type=int, default=16)
    data.add_argument(
        "--no-shuffle",
        action="store_true",
        help="disable training-set shuffling (the original permutation runs did this)",
    )

    model = parser.add_argument_group("model")
    model.add_argument("--model", default="leftnet", choices=sorted(MODELS))
    model.add_argument("--timesteps", type=int, default=5000)
    model.add_argument("--noise-schedule", default="cosine")

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--lr", type=float, default=5e-4)
    optim.add_argument("--batch-size", type=int, default=16)
    optim.add_argument("--epochs", type=int, default=2000)
    optim.add_argument("--limit-train-batches", type=int, default=200)
    optim.add_argument("--limit-val-batches", type=int, default=20)
    optim.add_argument("--eval-epochs", type=int, default=10)
    optim.add_argument("--seed", type=int, default=42)
    optim.add_argument("--ema", action="store_true")

    out = parser.add_argument_group("output")
    out.add_argument(
        "--project",
        default=None,
        help="wandb project; defaults to OAReactDiff_<dataset>[_subset<fraction>][_<permute>]",
    )
    out.add_argument("--version", default="0")
    out.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("checkpoint"),
        help="per-epoch checkpoints go to <root>/<project>/<run name>",
    )
    out.add_argument(
        "--final-checkpoint",
        type=Path,
        default=None,
        help="optional path for the end-of-training checkpoint",
    )
    out.add_argument(
        "--wandb-mode",
        default="offline",
        choices=["offline", "online", "disabled"],
    )
    return parser.parse_args()


def resolve_datadir(args: argparse.Namespace) -> Path:
    """Split directory for --dataset/--fraction, i.e. <data-root>/<stem>_splits or
    <stem>_subset_<fraction>_splits, unless --datadir overrides it."""
    if args.datadir is not None:
        return args.datadir
    stem = DATASET_DIRS[args.dataset]
    suffix = "_splits" if args.fraction == 100 else f"_subset_{args.fraction}_splits"
    return args.data_root / f"{stem}{suffix}"


def build_project_name(args: argparse.Namespace) -> str:
    """Wandb project and checkpoint-directory name, OAReactDiff_<dataset> plus
    _subset<fraction> and _<permute>_permuted suffixes where they apply."""
    if args.project is not None:
        return args.project
    name = f"OAReactDiff_{args.dataset}"
    if args.fraction != 100:
        name += f"_subset{args.fraction}"
    if args.permute != "none":
        name += f"_{args.permute.replace('+', 'n')}_permuted"
    return name


def build_configs(args: argparse.Namespace, datadir: Path) -> Tuple[dict, dict, dict]:
    model_config = MODEL_CONFIGS[args.model]
    optimizer_config = dict(
        lr=args.lr,
        betas=[0.9, 0.999],
        weight_decay=0,
        amsgrad=True,
    )
    training_config = dict(
        datadir=str(datadir) + os.sep,
        remove_h=False,
        bz=args.batch_size,
        num_workers=args.num_workers,
        clip_grad=True,
        gradient_clip_val=None,
        ema=args.ema,
        ema_decay=0.999,
        swapping_react_prod=True,
        append_frag=False,
        use_by_ind=True,
        reflection=False,
        single_frag_only=False,
        only_ts=False,
        lr_schedule_type=None,
        lr_schedule_config=dict(gamma=0.8, step_size=100),
        permute=args.permute,
        shuffle=not args.no_shuffle,
    )
    return model_config, optimizer_config, training_config


def build_module(
    args: argparse.Namespace,
    model_config: dict,
    optimizer_config: dict,
    training_config: dict,
) -> DDPMModule:
    """DDPM module over the fixed R/TS/P layout; no fragment is held fixed (fixed_idx=None)."""
    node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
    fragment_names: List[str] = ["R", "TS", "P"]
    norm_values: Tuple = (1.0, 1.0, 1.0)
    norm_biases: Tuple = (0.0, 0.0, 0.0)
    edge_cutoff: Optional[float] = None
    fixed_idx: Optional[List] = None
    return DDPMModule(
        model_config,
        optimizer_config,
        training_config,
        node_nfs,
        0,  # edge_nf
        1,  # condition_nf
        fragment_names,
        3,  # pos_dim
        True,  # update_pocket_coords
        True,  # condition_time
        edge_cutoff,
        norm_values,
        norm_biases,
        args.noise_schedule,
        args.timesteps,
        1e-5,  # precision
        "l2",  # loss_type
        True,  # pos_only
        "TS1x",  # process_type
        MODELS[args.model],
        None,  # enforce_same_encoding
        [1.0, 2.0, 1.0],  # scales
        source=None,
        fixed_idx=fixed_idx,
        eval_epochs=args.eval_epochs,
    )


def build_callbacks(ckpt_path: Path, training_config: dict) -> list:
    callbacks = [
        EarlyStopping(
            monitor="val-totloss",
            patience=2000,
            verbose=True,
            log_rank_zero_only=True,
        ),
        ModelCheckpoint(
            monitor="val-totloss",
            dirpath=str(ckpt_path),
            filename="ddpm-{epoch:03d}-{val-totloss:.2f}",
            every_n_epochs=1,
            save_top_k=1,
        ),
        TQDMProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]
    if training_config["ema"]:
        callbacks.append(EMACallback(decay=training_config["ema_decay"]))
    return callbacks


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    os.environ["WANDB_MODE"] = args.wandb_mode

    datadir = resolve_datadir(args)
    if not datadir.is_dir():
        raise FileNotFoundError(f"split directory not found: {datadir}")
    project = build_project_name(args)
    run_name = f"{args.model}-{args.version}-" + str(uuid4()).split("-")[-1]

    model_config, optimizer_config, training_config = build_configs(args, datadir)
    seed_everything(args.seed, workers=True)
    ddpm = build_module(args, model_config, optimizer_config, training_config)

    config = model_config.copy()
    config.update(optimizer_config)
    config.update(training_config)
    wandb_logger = WandbLogger(project=project, log_model=False, name=run_name)
    try:  # Avoid errors for creating wandb instances multiple times
        wandb_logger.experiment.config.update(config)
        wandb_logger.watch(ddpm.ddpm.dynamics, log="all", log_freq=100, log_graph=False)
    except Exception:
        pass

    ckpt_path = args.checkpoint_root / project / wandb_logger.experiment.name
    ckpt_path.mkdir(parents=True, exist_ok=True)
    model_source = Path(__file__).resolve().parent.parent / "model" / f"{args.model}.py"
    shutil.copy(model_source, ckpt_path / model_source.name)
    print("config: ", config)

    devices = list(range(torch.cuda.device_count()))
    strategy = DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else None
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        deterministic=False,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=1,
        callbacks=build_callbacks(ckpt_path, training_config),
        profiler=None,
        logger=wandb_logger,
        accumulate_grad_batches=1,
        gradient_clip_val=training_config["gradient_clip_val"],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(ddpm)
    if args.final_checkpoint is not None:
        args.final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.final_checkpoint))
    if trainer.logger is not None:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
