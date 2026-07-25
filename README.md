# PositionTS

Benchmark of machine-learning methods for 3D reaction transition-state (TS) prediction. Each method lives in its own directory and keeps its original code, README and licence.

| Directory | Method | Framework |
|---|---|---|
| `OAReactDiff/` | OA-ReactDiff — SE(3)-equivariant diffusion (DDPM, LEFTNet) | PyTorch Lightning |
| `react-ot/` | React-OT — Schrödinger-bridge / optimal-transport TS generator | PyTorch Lightning |
| `TS-GAN/` | TS-GAN — conditional GAN over Coulomb-matrix images | TensorFlow 2 (Keras) |
| `ts_gen/` | ts_gen — graph-to-coordinates GNN | TensorFlow 1.x |

Each method needs its own environment; their framework versions conflict.

---

## OA-ReactDiff

Setup follows the method's own README: `conda env create -f env.yaml && conda activate oa_reactdiff && pip install -e .`.

### Training

All experiments run from a single entry point, `oa_reactdiff/trainer/train.py` — dataset, supervision level and atom ordering are arguments, nothing is hardcoded.

```bash
cd OAReactDiff/oa_reactdiff/trainer

python train.py --dataset ts1x                          # full Transition1x
python train.py --dataset rgd1 --fraction 10            # 10% of RGD1
python train.py --dataset ts1x --permute product        # permuted product atom order
python train.py --help                                  # all options
```

| Argument | Values | Purpose |
|---|---|---|
| `--dataset` | `ts1x`, `iso`, `rgd1`, `rgdx`, `rgdy` | which benchmark |
| `--fraction` | `100`, `50`, `10` | supervision level |
| `--permute` | `none`, `product`, `ts`, `product+ts` | which fragments get randomly relabelled atoms |
| `--data-root` / `--datadir` | path | where the split directories live |
| `--model` | `leftnet`, `egnn` | backbone |
| `--project`, `--checkpoint-root`, `--final-checkpoint` | path / name | output locations |
| `--lr`, `--batch-size`, `--epochs`, `--timesteps`, `--seed`, `--num-workers`, `--eval-epochs`, `--limit-train-batches`, `--limit-val-batches`, `--ema`, `--no-shuffle`, `--wandb-mode` | | training knobs |

`--dataset`/`--fraction` resolve to `<data-root>/<stem>_splits/` or `<data-root>/<stem>_subset_<fraction>_splits/`, where `stem` is `transition1x`, `isomer`, `rgd1`, `rgdx` or `rgdy`. Each directory must contain `train_split.pkl` and `validation_split.pkl`. Reduced-supervision runs read a prebuilt subset directory rather than subsampling at runtime, so the 10% and 50% splits — including their validation sets — must be generated beforehand.

Per-epoch checkpoints go to `<checkpoint-root>/<project>/<run name>/`; `--project` defaults to `OAReactDiff_<dataset>[_subset<fraction>][_<permute>_permuted]`. Logging is wandb in `offline` mode by default.

`--permute` relabels atoms inside the dataset's `collate_fn`, which serves the training, validation and test dataloaders alike — so it permutes during training, not only at evaluation.

### Generating transition states

`oa_reactdiff/learnabilty/` holds one generation script per trained model — `generate_<dataset>.py` for the fully supervised checkpoint, `_s10` / `_s50` for the reduced-supervision ones:

| Dataset | 100% | 50% | 10% |
|---|---|---|---|
| Transition1x | `generate_ts1x.py` | `generate_ts1x_s50.py` | `generate_ts1x_s10.py` |
| Isomerization | `generate_iso.py` | `generate_iso_s50.py` | `generate_iso_s10.py` |
| RGD1 (`rgd1`) | `generate_rgd1.py` | `generate_rgd1_s50.py` | `generate_rgd1_s10.py` |
| RGD1 (`rgdx`) | `generate_rgdx.py` | `generate_rgdx_s50.py` | `generate_rgdx_s10.py` |
| RGD1 (`rgdy`) | `generate_rgdy.py` | `generate_rgdy_s50.py` | `generate_rgdy_s10.py` |

Each script loads its checkpoint, replaces the sampling schedule with 150 `polynomial_2` timesteps (training uses 5000 `cosine`), streams that dataset's **full** test split, and generates a transition state per reaction by inpainting: `inplaint_batch` fixes fragments 0 and 2 (reactant, product) and denoises fragment 1 (the TS), with `resamplings=5` and `jump_length=5`.

Two paths must be filled in before running — both are empty strings with the expected value documented inline:

```python
tspath = ""        # checkpoint root, e.g. "../trainer/checkpoint"
pkl_path = ""      # test split, e.g. "../data/rgdx_splits/test_split.pkl"
```

`tspath` is the `--checkpoint-root` used at training time; the entry consumed from the `checkpoints` dict is `leftnet_actual_test_best[_<dataset>]`. Only `generate_ts1x.py` ships a working relative `pkl_path`.

Output goes to `tmp_<dataset>[_s10|_s50]/entry_<index>.xyz` — one file per reaction, containing the generated TS geometry only. Indices are global across batches, which is why the loader runs with `shuffle=False`; re-running overwrites in place.

These scripts compute **no metrics** — the RMSD and energy imports are deliberately commented out. Median RMSD / median absolute energy error are computed separately from the emitted `.xyz` files; see `oa_reactdiff/analyze/rmsd.py` and `oa_reactdiff/evaluate/`.

Run them through the same launcher as training:

```bash
cd OAReactDiff/oa_reactdiff/learnabilty
USE_SRUN=0 sbatch ../run.slurm oa_reactdiff generate_rgdx_s10.py
```

### Upstream evaluation scripts

`oa_reactdiff/evaluate/` is unmodified upstream code: `evaluate_ts_w_rp.py` (TS inpainting + RMSD), `evaluate_rmsd_vs_ediff.py` (RMSD against DFT energy difference), `generate_confidence_sample.py`, two `run_*.sh` wrappers, and `utils.py`, which provides the `set_new_schedule` / `inplaint_batch` / `batch_ts_deltaE` helpers the generation scripts import. The `evaluate_*.py` entry points reference upstream absolute checkpoint paths and need editing before use.

### Running on SLURM

`oa_reactdiff/run.slurm` is a single launcher for both training and evaluation:

```bash
sbatch [SBATCH_OPTIONS] <path>/run.slurm CONDA_ENV PYTHON_SCRIPT [PYTHON_ARGS...]
```

Submit from the directory containing the Python script; the job runs in `$SLURM_SUBMIT_DIR`.

```bash
cd OAReactDiff/oa_reactdiff/trainer
sbatch ../run.slurm oa_reactdiff train.py --dataset rgd1                     # cluster default queue, 48h
sbatch --partition=<partition> --qos=<qos> ../run.slurm oa_reactdiff train.py --dataset iso
sbatch --partition=<partition> -t 120:00:00 ../run.slurm oa_reactdiff train.py --dataset rgdy

cd ../learnabilty
USE_SRUN=0 sbatch ../run.slurm oa_reactdiff generate_rgd1.py
```

The script requests `-N 1 --ntasks-per-node=1 --gres=gpu:1 --mem-per-cpu=60G -t 48:00:00`. Partition and QoS are site-specific and deliberately unset, so jobs go to the cluster default queue; pass `--partition` / `--qos` per submission, since `sbatch` command-line options take precedence over the in-file directives. `USE_SRUN=0` calls `python` directly instead of through `srun`. Output goes to `<script>_<jobid>_output.log` in the submit directory.

### Notes

- `oa_reactdiff/trainer/train_ts1x.py` and `train_confidence_ts1x.py` are the unmodified upstream entry points.
- `oa_reactdiff/trainer/train_rgd1_with_bugged_rmsd.py` is retained deliberately: it labels a run made against the buggy RMSD implementation. The label is the only thing that selects it — the RMSD code is chosen by import, not by the training script.
- Two RMSD implementations exist. `oa_reactdiff/analyze/rmsd.py` is the one every trainer imports; it clamps values with `min(rmsd, 1.0)`, falls back to `1.0` on exception, silently returns the matcher `threshold` (`0.5`) when `GeneticOrderMatcher` finds no pair, and passes `ignore_chirality=True`. `oa_reactdiff/analyze/rmsd_corrected.py` removes the clamp, applies a `sqrt(3)` factor, uses `argmin` over candidates and sets `ignore_chirality=False` — but nothing imports it, so reproducing corrected numbers means swapping the module.
