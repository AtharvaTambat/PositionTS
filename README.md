# PositionTS

Code for the transition-state prediction experiments. The six baselines live in `baselines/`, each retaining its upstream licence and requiring its own environment. Shared evaluation code is in `evaluation/`.

## Job submission

```bash
sbatch [SBATCH_OPTIONS] <path>/run.slurm CONDA_ENV PYTHON_SCRIPT [ARGS...]
```

Submit from the directory containing the script. Partition and QoS are unset; supply them per submission. Toggles: `USE_SRUN=0`, `LOG_TO_FILE=0`, `CONDA_SH=<path>`.

## OA-ReactDiff

```bash
cd baselines/OAReactDiff/oa_reactdiff/trainer
python train.py --dataset rgd1 --fraction 10
```

| Argument | Values |
|---|---|
| `--dataset` | `ts1x`, `iso`, `rgd1`, `rgdx`, `rgdy` |
| `--fraction` | `100`, `50`, `10` (reduced supervision) |
| `--permute` | `none`, `product`, `ts`, `product+ts` (atom-order robustness) |
| `--data-root` | location of the split directories |

Splits: `<data-root>/<stem>_splits/` and `<stem>_subset_<fraction>_splits/`, each containing `train_split.pkl` and `validation_split.pkl`.

Permutation is applied in `collate_fn`, during training as well as evaluation. Reported settings: `--permute product --no-shuffle`, `--permute ts --no-shuffle`, `--permute product+ts`.

Generation: `baselines/OAReactDiff/oa_reactdiff/learnabilty/generate_<dataset>[_s10|_s50].py`, with `tspath` and `pkl_path` to be set. Writes `tmp_<dataset>*/entry_<i>.xyz`.

`baselines/OAReactDiff/oa_reactdiff/analyze/rmsd.py` clamps RMSD at 1.0 and sets `ignore_chirality=True`. `rmsd_corrected.py` alongside it is unused.

## React-OT

```bash
cd baselines/react-ot/reactot/trainer
python train.py --dataset rgd1 --fraction 10 --data-root <path>
```

`--dataset` and `--fraction` as above; `--data-root` is required. Split layout as above; `SBModule.setup` reads `train_split.pkl` and `validation_split.pkl`.

| Generation directory | Experiment |
|---|---|
| `baselines/react-ot/reactot/evaluate/` | main results |
| `baselines/react-ot/reactot/learnability/` | reduced supervision |
| `baselines/react-ot/reactot/permutation/` | atom-order robustness |
| `baselines/react-ot/reactot/generalize/trained_<rgd1\|rgdx\|rgdy>/` | cross-regime transfer |

Each requires `tspath` and `pkl_path`. `Trainer(replace_sampler_ddp=False)` requires Lightning < 2.0.

## TS-GAN

```bash
cd baselines/TS-GAN
sbatch ./run.slurm tsgan train.py --data_path ./data/rgd1_numpy/train/train.npy \
    --trained_model_path ./trained_models/rgd1
```

`create_dataset.sh` builds the Coulomb-matrix `.npy` inputs; `run.sh` and `run_generate_ts.sh` hold the training and prediction commands. Paths come from `DATA_ROOT`, `PERMUTED_ROOT`, `MODEL_ROOT`, `OUTPUT_ROOT`.

`predict.py` takes the generator checkpoint as an argument.

## ts_gen

```bash
cd baselines/ts_gen
sbatch ./run.slurm tsgen train.py --r_file <reactants.sdf> --p_file <products.sdf> \
    --ts_file <ts.sdf> --hidden_size 256
sbatch ./run.slurm tsgen predict_new.py --reactants … --products … --checkpoint … --output_dir …
```

Commands are recorded in `run.sh` and `run_generate_ts.sh`. `predict_new.py` supersedes `predict.py`.

## NeuralNEB

PaiNN potential inside a nudged-elastic-band search.

```bash
cd baselines/neuralneb
python scripts/download_qm9x.py data
sbatch ./run.slurm oa2 scripts/train.py
./run_all.sh rgd1
./run_all.sh transition1x --permuted
```

`run_all.sh` submits one job per reaction under a concurrency cap. Overrides: `DATA_ROOT`, `PERMUTED_ROOT`, `OUTPUT_ROOT`, `CONDA_ENV`, `PARTITION`, `MAX_JOBS`, `WALLTIME`.

## LearnTS

```bash
git submodule update --init baselines/learnTS/ard_gsm
cd baselines/learnTS/example
./run_train.sh
sbatch --mem=60G ../run.slurm learnts inference_modified.py
```

`run_train.sh` carries the reported hyperparameters. Set the checkpoint in `inference_modified.py` and `--directory` in `baselines/learnTS/learnts/db_ts1x.py`, which also builds the pickled databases.

## evaluation

| Script | Role |
|---|---|
| `evaluate.py`, `evaluate2.py`, `evaluate3.py`, `evaluate_multiple.py` | RMSD and energy error over generated structures |
| `evaluate_shuffle_ts.py` | atom-order comparison; writes `<dataset>.json` (the checked-in copies are in `ts_shuffling/`) |
| `get_median.py`, `metrics.py` | medians from one log; two-log minimum with mean, std and RMSE |
| `rmsd.py` | `rmsd_core`, `dmae_core` |
| `generate_ts.py`, `D2C.py` | coordinates from a predicted distance matrix |
| `geomopt.py` | energies, forces, Hessians, negative-eigenvalue counts |
| `flagger.py` | screens generated `.xyz` for implausible geometries |
| `shuffle_TS.py`, `make_permuted_pkl.py`, `toxify.py` | permuted split construction |
| `mw.py`, `n_heavy_atoms.py`, `xyz2mol.py` | molecular weight and heavy-atom counts |
| `gen_ts.sh` | recorded generation commands |

Arguments are positional. The four `evaluate*.py` variants assume different generated-file naming and infer the DFT level of theory differently from the output-folder name.
