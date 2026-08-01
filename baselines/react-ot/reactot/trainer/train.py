"""Train React-OT on any benchmark dataset and supervision level.

Replaces the previous one-script-per-configuration layout; everything that used to
be hardcoded is a command-line argument. See --help for the full list.
"""
import argparse
import os
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

from reactot.model import LEFTNet
from reactot.trainer.ema import EMACallback
from reactot.trainer.pl_trainer import DDPMModule, SBModule

DATASET_DIRS = {
    "ts1x": "transition1x",
    "iso": "isomer",
    "rgd1": "rgd1",
    "rgdx": "rgdx",
    "rgdy": "rgdy",
}
MODELS = {"leftnet": LEFTNet}
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


class OPT:
    """Solver settings consumed by the Schrodinger-bridge sampler."""

    def __init__(self, solver, method, atol=1e-2, rtol=1e-2):
        self.solver = solver
        self.method = method
        self.atol = atol
        self.rtol = rtol


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
    data.add_argument("--num-workers", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--model", default="leftnet", choices=sorted(MODELS))
    model.add_argument("--timesteps", type=int, default=3000)

    bridge = parser.add_argument_group("schrodinger bridge")
    bridge.add_argument("--mapping", default="R+P->TS")
    bridge.add_argument(
        "--mapping-initial",
        default="RP",
        help='"RP" for (r+p)/2, "GUESS" to start from a guess',
    )
    bridge.add_argument("--nfe", type=int, default=25)
    bridge.add_argument("--beta-max", type=float, default=0.3)
    bridge.add_argument("--power", type=float, default=0.5)
    bridge.add_argument("--inv-power", type=float, default=1.0)
    bridge.add_argument("--sigma", type=float, default=0.0)
    bridge.add_argument("--ts-guess", default=None)
    bridge.add_argument("--no-ot-ode", action="store_true")
    bridge.add_argument("--solver", default="ddpm")
    bridge.add_argument("--method", default="midpoint")

    optim = parser.add_argument_group("optimization")
    optim.add_argument("--lr", type=float, default=1e-4)
    optim.add_argument("--batch-size", type=int, default=14)
    optim.add_argument("--epochs", type=int, default=2000)
    optim.add_argument("--limit-train-batches", type=int, default=200)
    optim.add_argument("--limit-val-batches", type=int, default=20)
    optim.add_argument("--eval-epochs", type=int, default=1)
    optim.add_argument("--save-epochs", type=int, default=1)
    optim.add_argument("--seed", type=int, default=42)
    optim.add_argument("--no-ema", action="store_true")
    optim.add_argument(
        "--pretrain-checkpoint",
        type=Path,
        default=None,
        help="DDPM checkpoint to warm-start the model and encoders from",
    )

    out = parser.add_argument_group("output")
    out.add_argument(
        "--project",
        default=None,
        help="wandb project; defaults to ReactOT_<dataset>[_subset<fraction>]",
    )
    out.add_argument("--version", default="ts_guess_NEBCI-xtb-ema")
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
        "--wandb-mode", default="offline", choices=["offline", "online", "disabled"]
    )
    return parser.parse_args()


def resolve_datadir(args: argparse.Namespace) -> Path:
    """Split directory implied by --dataset/--fraction (<stem>_splits or <stem>_subset_<n>_splits), unless --datadir overrides it."""
    if args.datadir is not None:
        return args.datadir
    stem = DATASET_DIRS[args.dataset]
    suffix = "_splits" if args.fraction == 100 else f"_subset_{args.fraction}_splits"
    return args.data_root / f"{stem}{suffix}"


def build_project_name(args: argparse.Namespace) -> str:
    if args.project is not None:
        return args.project
    name = f"ReactOT_{args.dataset}"
    if args.fraction != 100:
        name += f"_subset{args.fraction}"
    return name


def build_configs(args: argparse.Namespace, datadir: Path) -> Tuple[dict, dict, dict]:
    """The model config returned is always LEFTNET_CONFIG, regardless of --model."""
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
        ema=not args.no_ema,
        ema_decay=0.999,
        swapping_react_prod=False,
        append_frag=False,
        use_by_ind=True,
        reflection=False,
        single_frag_only=False,
        only_ts=False,
        lr_schedule_type=None,
        lr_schedule_config=dict(gamma=0.8, step_size=10),
        use_sampler=True,
        sampler_config=dict(
            max_num=2800,  # This is for 16GB GPU; Scale linearly with memory
            mode="node^2",
            shuffle=True,
            ddp=False,
        ),
    )
    return LEFTNET_CONFIG, optimizer_config, training_config


def load_pretrained_source(checkpoint: Path, training_config: dict) -> dict:
    """Return the model, encoder and decoder state dicts of a DDPM checkpoint.

    Also records the checkpoint path and use_pretrain into training_config in place.
    """
    ddpm_trainer = DDPMModule.load_from_checkpoint(
        checkpoint_path=str(checkpoint),
        map_location="cpu",
    )
    training_config.update(
        {"checkpoint_path": str(checkpoint), "use_pretrain": True}
    )
    return {
        "model": ddpm_trainer.ddpm.dynamics.model.state_dict(),
        "encoders": ddpm_trainer.ddpm.dynamics.encoders.state_dict(),
        "decoders": ddpm_trainer.ddpm.dynamics.decoders.state_dict(),
    }


def build_module(
    args: argparse.Namespace,
    model_config: dict,
    optimizer_config: dict,
    training_config: dict,
    source: Optional[dict],
) -> SBModule:
    """Build the bridge module over the fixed R/TS/P fragment layout, with R and P held fixed."""
    node_nfs: List[int] = [9] * 3  # 3 (pos) + 5 (cat) + 1 (charge)
    fragment_names: List[str] = ["R", "TS", "P"]
    norm_values: Tuple = (1.0, 1.0, 1.0)
    norm_biases: Tuple = (0.0, 0.0, 0.0)
    edge_cutoff: Optional[float] = None
    return SBModule(
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
        "cosine",  # noise_schedule, not used by the bridge
        args.timesteps,
        1e-5,  # precision, not used
        "l2",  # loss_type
        True,  # pos_only
        "TS1x",  # process_type
        MODELS[args.model],
        None,  # enforce_same_encoding
        [1.0, 2.0, 1.0],  # scales
        source=source,
        fixed_idx=[0, 2],
        eval_epochs=args.eval_epochs,
        mapping=args.mapping,
        mapping_initial=args.mapping_initial,
        nfe=args.nfe,
        beta_max=args.beta_max,
        ot_ode=not args.no_ot_ode,
        power=args.power,
        inv_power=args.inv_power,
        sigma=args.sigma,
        ts_guess=args.ts_guess,
    )


def build_callbacks(args: argparse.Namespace, ckpt_path: Path, module: SBModule) -> list:
    """Lightning callbacks for the run; both early stopping and checkpointing track val_ep_scaled_err."""
    callbacks = [
        EarlyStopping(
            monitor="val_ep_scaled_err",
            patience=2000,
            verbose=True,
            log_rank_zero_only=True,
        ),
        ModelCheckpoint(
            monitor="val_ep_scaled_err",
            dirpath=str(ckpt_path),
            filename="sb-{epoch:03d}-{val_ep_scaled_err:.4f}",
            every_n_epochs=args.save_epochs,
            save_top_k=1,
        ),
        TQDMProgressBar(),
        LearningRateMonitor(logging_interval="step"),
    ]
    if not args.no_ema:
        callbacks.append(EMACallback(pl_module=module, decay=0.999))
    return callbacks


def main() -> None:
    args = parse_args()
    os.environ["WANDB_MODE"] = args.wandb_mode

    datadir = resolve_datadir(args)
    if not datadir.is_dir():
        raise FileNotFoundError(f"split directory not found: {datadir}")
    project = build_project_name(args)
    run_name = f"{args.model}-{args.version}-" + str(uuid4()).split("-")[-1]

    model_config, optimizer_config, training_config = build_configs(args, datadir)
    source = None
    if args.pretrain_checkpoint is not None:
        source = load_pretrained_source(args.pretrain_checkpoint, training_config)

    seed_everything(args.seed, workers=True)
    ddpm = build_module(args, model_config, optimizer_config, training_config, source)
    ddpm.ddpm.opt = OPT(solver=args.solver, method=args.method)

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
    print("config: ", config)

    devices = list(range(torch.cuda.device_count()))
    strategy = DDPStrategy(find_unused_parameters=True) if len(devices) > 1 else None
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        deterministic=False,
        devices=devices,
        strategy=strategy,
        log_every_n_steps=20,
        callbacks=build_callbacks(args, ckpt_path, ddpm),
        profiler=None,
        logger=wandb_logger,
        accumulate_grad_batches=1,
        gradient_clip_val=training_config["gradient_clip_val"],
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        replace_sampler_ddp=False,
    )

    trainer.fit(ddpm)
    if args.final_checkpoint is not None:
        args.final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(str(args.final_checkpoint))
    if trainer.logger is not None:
        wandb_logger.experiment.finish()


if __name__ == "__main__":
    main()
