# PositionTS

Benchmark of machine-learning methods for 3D reaction transition-state (TS) prediction. Each method lives in its own directory and keeps its original code, README and licence.

| Directory | Method | Framework |
|---|---|---|
| `OAReactDiff/` | OA-ReactDiff — SE(3)-equivariant diffusion (DDPM, LEFTNet) | PyTorch Lightning |
| `react-ot/` | React-OT — Schrödinger-bridge / optimal-transport TS generator | PyTorch Lightning |
| `TS-GAN/` | TS-GAN — conditional GAN over Coulomb-matrix images | TensorFlow 2 (Keras) |
| `ts_gen/` | ts_gen — graph-to-coordinates GNN | TensorFlow 1.x |
| `neuralneb/` | NeuralNEB — PaiNN potential inside a nudged-elastic-band search | PyTorch |
| `evaluation/` | shared evaluation, metric and dataset-analysis scripts (not a baseline) | — |

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
| `--lr`, `--batch-size`, `--epochs`, `--timesteps`, `--noise-schedule`, `--seed`, `--num-workers`, `--eval-epochs`, `--limit-train-batches`, `--limit-val-batches`, `--ema`, `--no-shuffle`, `--version`, `--wandb-mode` | | training knobs |

`--dataset`/`--fraction` resolve to `<data-root>/<stem>_splits/` or `<data-root>/<stem>_subset_<fraction>_splits/`, where `stem` is `transition1x`, `isomer`, `rgd1`, `rgdx` or `rgdy`. Each directory must contain `train_split.pkl` and `validation_split.pkl`. Reduced-supervision runs read a prebuilt subset directory rather than subsampling at runtime, so the 10% and 50% splits — including their validation sets — must be generated beforehand.

Per-epoch checkpoints go to `<checkpoint-root>/<project>/<run name>/`; `--project` defaults to `OAReactDiff_<dataset>[_subset<fraction>][_<permute>_permuted]`, with `+` rewritten as `n` (so `product+ts` yields `_productnts_permuted`). Logging is wandb in `offline` mode by default.

`--permute` relabels atoms inside the dataset's `collate_fn`, which serves the training, validation and test dataloaders alike — so it permutes during training, not only at evaluation.

The original per-permutation scripts did not agree on shuffling, so reproducing them needs different flags: the product-only and TS-only forks disabled training-set shuffling, while the product+ts fork shuffled through a `DistributedSampler`. Use `--permute product --no-shuffle`, `--permute ts --no-shuffle`, and `--permute product+ts` (shuffling left on).

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

`tspath` is the `--checkpoint-root` used at training time; the entry consumed from the `checkpoints` dict is `leftnet_actual_test_best[_<dataset>]`. Twelve of the fifteen scripts leave `pkl_path` empty; the three Transition1x ones (`generate_ts1x.py`, `_s10`, `_s50`) ship a working relative path.

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

- `oa_reactdiff/trainer/train_confidence_ts1x.py` is the unmodified upstream entry point. `train_ts1x.py` is upstream's script **with retuned hyperparameters**: `lr` 2.5e-4 → 5e-4, `bz` 14 → 16, `single_frag_only` True → False, `eval_epochs` 10 → 1, `save_top_k` -1 → 1, DDP forced on, and wandb set to offline. The consolidated `train.py` inherits these values, so reported OA-ReactDiff numbers are not upstream defaults.
- `oa_reactdiff/trainer/train_rgd1_with_bugged_rmsd.py` is retained deliberately: it labels a run made against the buggy RMSD implementation. The label is the only thing that selects it — the RMSD code is chosen by import, not by the training script.
- `pretrained-ts1x-diff.ckpt`, which upstream tracks at the `OAReactDiff/` root, is absent here, as are upstream's three `oa_reactdiff/data/transition1x/*.pkl`; `demo.py` and `OA-ReactDiff.ipynb` need them.
- Two RMSD implementations exist. `oa_reactdiff/analyze/rmsd.py` is the one every trainer imports; it clamps values with `min(rmsd, 1.0)`, falls back to `1.0` on exception, silently returns the matcher `threshold` (`0.5`) when `GeneticOrderMatcher` finds no pair, and passes `ignore_chirality=True`. `oa_reactdiff/analyze/rmsd_corrected.py` removes the clamp, applies a `sqrt(3)` factor, uses `argmin` over candidates and sets `ignore_chirality=False` — but nothing imports it, so reproducing corrected numbers means swapping the module.

---

## React-OT

Setup follows the method's own README: `conda env create -f env.yaml && conda activate reactot && pip install -e .`. Ignore the top-level `requirements.txt`, which is unrelated.

### Training

All experiments run from `reactot/trainer/train.py`; dataset and supervision level are arguments.

```bash
cd react-ot/reactot/trainer

python train.py --dataset ts1x                       # full Transition1x
python train.py --dataset rgd1 --fraction 10         # 10% of RGD1
python train.py --help                               # all options
```

| Argument | Values | Purpose |
|---|---|---|
| `--dataset` | `ts1x`, `iso`, `rgd1`, `rgdx`, `rgdy` | which benchmark |
| `--fraction` | `100`, `50`, `10` | supervision level |
| `--data-root` / `--datadir` | path | where the split directories live |
| `--mapping`, `--mapping-initial`, `--nfe`, `--beta-max`, `--power`, `--inv-power`, `--sigma`, `--ts-guess`, `--no-ot-ode`, `--solver`, `--method` | | Schrödinger-bridge settings |
| `--model` | `leftnet` | backbone |
| `--pretrain-checkpoint` | path | warm-start model/encoders from a DDPM checkpoint |
| `--project`, `--version`, `--checkpoint-root`, `--final-checkpoint` | path / name | output locations |
| `--lr`, `--batch-size`, `--epochs`, `--timesteps`, `--seed`, `--num-workers`, `--eval-epochs`, `--save-epochs`, `--limit-train-batches`, `--limit-val-batches`, `--no-ema`, `--wandb-mode` | | training knobs |

`--dataset`/`--fraction` resolve to `<data-root>/<stem>_splits/` or `<data-root>/<stem>_subset_<fraction>_splits/`, with `stem` in `transition1x`, `isomer`, `rgd1`, `rgdx`, `rgdy`. Each directory must hold `train_split.pkl` and `validation_split.pkl`. Note that `SBModule.setup` in this fork reads those filenames rather than upstream's `train_rpsb_all.pkl` / `valid_rpsb_all.pkl`. The original runs read the splits from the OA-ReactDiff data tree, so `--data-root` must be set: its default (`reactot/data`) does not exist and the script exits with `FileNotFoundError`.

Per-epoch checkpoints go to `<checkpoint-root>/<project>/<run name>/`, selected on `val_ep_scaled_err`; `--project` defaults to `ReactOT_<dataset>[_subset<fraction>]`. EMA is on by default (`--no-ema` disables it).

### Generating transition states

Four directories hold generation scripts, one per experiment, each pinned to a particular checkpoint and split:

| Directory | Purpose |
|---|---|
| `reactot/evaluate/` | main results — `generate_<dataset>.py`, plus `evaluate_res2_<dataset>.py` which also computes DFT energy differences |
| `reactot/learnability/` | reduced supervision — `generate_<dataset>.py` and `generate_<dataset>_s10.py` |
| `reactot/permutation/` | atom-order robustness — reads a permuted copy of the test split |
| `reactot/generalize/trained_<rgd1\|rgdx\|rgdy>/` | cross-regime transfer — a model trained on one RGD1 subset generating for the others |

Each needs two paths filled in, both left as empty strings with the expected value documented inline:

```python
tspath = ""        # checkpoint dir incl. project subdir, e.g. "../trainer/checkpoint/ReactOT_rgd1/"
pkl_path = ""      # test split, e.g. "../data/rgd1_splits/test_split.pkl"
```

Output goes to a per-experiment directory (`tmp_<dataset>/`, `s10_<dataset>/`, …) as `entry_<index>.xyz`.

### Running on SLURM

`reactot/run.slurm` is the single launcher, same contract as the OA-ReactDiff one:

```bash
cd react-ot/reactot/trainer
sbatch ../run.slurm reactot train.py --dataset rgd1
sbatch --partition=<partition> --qos=<qos> ../run.slurm reactot train.py --dataset iso

cd ../evaluate
USE_SRUN=0 sbatch ../run.slurm reactot generate_rgd1.py
```

Partition and QoS are unset by design; pass them per submission. `USE_SRUN=0` skips `srun`, which is what the generation and evaluation scripts did.

### Notes

- `reactot/trainer/train_rpsb_ts1x.py` is the unmodified upstream entry point.
- `reactot/trainer/train_rpsb_rgd1_with_bugged_rmsd.py` is retained deliberately, mirroring the OA-ReactDiff case: it labels a run made against the buggy RMSD implementation (`reactot/analyze/rmsd.py` versus the unimported `rmsd_corrected.py`).
- `Trainer(replace_sampler_ddp=False)` is preserved as the author ran it. That keyword was removed in PyTorch Lightning 2.0, so it requires a pre-2.0 Lightning despite `env.yaml` pinning 2.4.0.
- `reactot-pretrained.ckpt`, which upstream tracks at the repository root, is not present here; `evaluation.py` and `appmain.py` need it.

---

## TS-GAN

Training, dataset construction and prediction are already argument-driven, so there is one script per task rather than one per configuration. `TS-GAN/run.slurm` is the single SLURM launcher, replacing the previous `script_{a40,l40,dgx}.slurm` copies; partition and QoS are unset by design.

```bash
cd TS-GAN
sbatch ./run.slurm tsgan train.py --data_path ./data/rgd1_numpy/train/train.npy \
    --trained_model_path ./trained_models/rgd1   # required: without it the first checkpoint write raises TypeError
sbatch --partition=<partition> --qos=<qos> ./run.slurm tsgan predict.py ...
```

Three shell files record the command sequences used. The absolute paths in them were replaced by environment variables, defaulted at the top of each file and overridable from the shell. Paths that were already repository-relative were left as they were, so not every path in every file is variable-driven — `run.sh` uses none of the four:

```bash
DATA_ROOT=…       # dataset splits (shared with the OA-ReactDiff experiments)
PERMUTED_ROOT=…   # atom-order-permuted copies of the splits
MODEL_ROOT=…      # trained checkpoints
OUTPUT_ROOT=…     # generated transition states
```

| File | Purpose |
|---|---|
| `create_dataset.sh` | `convert_pkl_to_xyx.py` → `gen_data.py`: split pickles to `.xyz`, then to a Coulomb-matrix `.npy`. `convert_pkl_to_xyx_permd.py` writes the permuted copies. |
| `run.sh` | training runs, one line per dataset and supervision level |
| `run_generate_ts.sh` | prediction runs over the test splits |

`g_model.h5` and `d_model.h5` at the repository root are the authors' own trained GAN weights (they match none of upstream's copies under `test_cases/`, which is absent from this tree). `predict.py` takes its generator checkpoint as a command-line argument rather than loading these, and `d_model.h5` is only ever written, by `train.py` — neither is read by any script as it stands.

## ts_gen

`train.py` and `predict_new.py` are argparse-driven, so no per-configuration copies exist. `ts_gen/run.slurm` is the single launcher (the shell recipes previously referenced `script_*.slurm` files that were never present in this directory).

```bash
cd ts_gen
sbatch ./run.slurm tsgen train.py --r_file … --p_file … --ts_file … --hidden_size 256
sbatch ./run.slurm tsgen predict_new.py --reactants … --products … --checkpoint … --output_dir …
```

`run.sh` records the training runs and `run_generate_ts.sh` / `run_generate_ts_copy.sh` the prediction runs, using the same `DATA_ROOT` / `PERMUTED_ROOT` / `MODEL_ROOT` / `OUTPUT_ROOT` variables. Both generate recipes are kept: they are not duplicates, differing in which variant is live and in output naming (`isomer_perm` versus `isomer_permd`).

`use_trained_model.ipynb` needs `model_path` filled in — left as an empty string with the expected value documented in the cell above it.

### Notes

- `predict.py` is superseded by `predict_new.py`, which is what every shell recipe invokes.
- Upstream's `data/` (five SDFs) and `log/2layers_256hs_3its/` (the pretrained checkpoint) are absent from this tree, so training data and the shipped model must be obtained from upstream.
- The notebook's stored outputs come from upstream's original run (TensorFlow deprecation warnings); the home-directory prefix in them was replaced with `<conda>`, and an author-added kernel-crash error output was removed.

---

## NeuralNEB

Unlike the generative baselines, NeuralNEB does not learn a transition state directly: it trains a PaiNN interatomic potential and then runs a nudged-elastic-band search per reaction. Upstream is `gitlab.com/matschreiner/neuralneb` (Mathias Schreiner); install with `pip install -e .`.

```bash
cd neuralneb
python scripts/download_qm9x.py data          # fetch qm9x.db
sbatch ./run.slurm oa2 scripts/train.py       # train the PaiNN potential
```

`run.slurm` is the single launcher, replacing five near-identical wrappers (`script_{a40,l40,dgx}.slurm` and two `_logging` variants). Beyond the shared `USE_SRUN`, it takes:

- `LOG_TO_FILE=0` — stream output to the job's stdout instead of `<script>_<jobid>_output.log`. This is what the `_logging` suffix used to select.
- `CONDA_SH=<path>` — source a specific conda profile script, replacing a hardcoded absolute path to one user's miniconda install.

### Running the NEB search over a dataset

One NEB job per reaction, throttled so only a few queue at once:

```bash
cd neuralneb
./run_all.sh rgd1                     # full test split
./run_all.sh transition1x --permuted  # atom-order-permuted split
PARTITION=<queue> MAX_JOBS=8 ./run_all.sh rgdy
```

This replaces four copies of the same 82-line submitter that differed only in three path lines. `<dataset>` is `transition1x`, `isomer`, `rgd1`, `rgdx` or `rgdy`; `--permuted` switches from `$DATA_ROOT/<dataset>/xyz_files/test_split/` to `$PERMUTED_ROOT/<dataset>_splits/xyz_data/` and appends `_permd` to the output directory. `DATA_ROOT`, `PERMUTED_ROOT`, `OUTPUT_ROOT`, `CONDA_ENV`, `PARTITION` and `MAX_JOBS` are all environment-overridable. Progress is appended to `progress_<dataset>.log`, and SLURM's own `.out`/`.err` files are suppressed since `run.slurm` writes its own log.

`scripts/` holds three NEB drivers — `neb.py` takes two `.xyz` files positionally, `neb_new.py` takes `--reactant_dir`/`--product_dir`, and `neb_new_2.py` is the one `run_all.sh` submits. They are genuinely different implementations, not copies.

## evaluation

Shared scripts that turn generated geometries into the numbers reported in the paper. Not a baseline, so it has no upstream. `evaluation/run.slurm` is its single launcher (replacing three wrappers plus two more under `generalization/tsgan/trained_rgdx/`).

```bash
cd evaluation
sbatch ./run.slurm oa2 evaluate.py
```

| Script | Role |
|---|---|
| `evaluate.py`, `evaluate2.py`, `evaluate3.py`, `evaluate_multiple.py` | RMSD and energy error over a directory of generated structures; the numbered variants differ in which inputs and reference sets they read |
| `evaluate_shuffle_ts.py` | the atom-order-permutation comparison |
| `metrics.py`, `get_median.py` | parse a run's log file and report medians |
| `rmsd.py` | `rmsd_core` / `dmae_core` — the RMSD and D-MAE implementations |
| `generate_ts.py`, `D2C.py` | reconstruct 3D coordinates from a predicted distance/Coulomb matrix |
| `geomopt.py` | saddle-point optimisation, Hessian and negative-eigenvalue counting |
| `flagger.py` | screens generated `.xyz` files for implausible geometries |
| `shuffle_TS.py`, `make_permuted_pkl.py`, `toxify.py` | build the permuted copies of a test split |
| `mw.py`, `calculate_mw.py`, `n_heavy_atoms.py`, `xyz2mol.py` | dataset analytics — molecular weight and heavy-atom counts |
| `ts_shuffling/*.json` | per-dataset permutation records, keyed by `entry_<n>.xyz` |
| `gen_ts.sh` | the recorded generation commands, with paths via `DATA_ROOT` / `INFERENCE_ROOT` |

Note `metrics.py` and `get_median.py` extract a path from the log text with a `/home/...` regex, so they expect logs produced on the original cluster layout.
