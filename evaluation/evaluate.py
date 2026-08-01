"""RMSD, D-MAE and energy error between reference and generated transition states.

Replaces evaluate.py, evaluate2.py, evaluate3.py and evaluate_multiple.py, which
differed only in how the generated filename was built and how the level of theory
was guessed from the output-folder name. The level of theory is now given by
--datatype rather than parsed out of a path.
"""
import argparse
import os
import sys

import numpy as np

from geomopt import calc_deltaE
from rmsd import rmsd_n_dmae

QM_KWDS = {
    "transition1x": ["wb97x", "631g*", False],
    "isomer": ["wb97x", "def2-TZVP", True],
    "rgd1": ["b3lyp", "def2-TZVP", True],
    "rgdx": ["b3lyp", "def2-TZVP", True],
    "rgdy": ["b3lyp", "def2-TZVP", True],
    "rgdx10": ["b3lyp", "def2-TZVP", True],
}


def parse_args() -> argparse.Namespace:
    """Parse the command line, rejecting a {fold} pattern that is not accompanied by --folds."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""patterns:
  '{stem}_ts.xyz'             <reference stem> + _ts.xyz
  'tsgen_idx_{index}.xyz'     index taken from the reference filename's trailing field
  'fold_{fold}_{stem}.xyz'    one file per fold; requires --folds
""",
    )
    parser.add_argument("gt_folder")
    parser.add_argument("gen_folder")
    parser.add_argument("--datatype", required=True, choices=sorted(QM_KWDS))
    parser.add_argument("--pattern", default="{stem}_ts.xyz")
    parser.add_argument("--folds", type=int, default=0)
    args = parser.parse_args()
    if "{fold}" in args.pattern and args.folds < 1:
        parser.error("--folds is required when the pattern contains {fold}")
    return args


def reference_index(filename: str) -> int:
    """Trailing underscore-separated field of a reference filename, as an integer."""
    return int(filename.split("_")[-1].split(".")[0])


def generated_name(pattern: str, filename: str, fold: int) -> str:
    """Name of the generated file for a reference file, filling {stem}, {index} and {fold}."""
    fields = {"stem": filename.split(".")[0], "fold": fold}
    if "{index}" in pattern:
        fields["index"] = reference_index(filename)
    return pattern.format(**fields)


def atom_count(path: str) -> int:
    with open(path, "r") as f:
        return int(f.readline().strip())


def summarise(prefix: str, delEs: list, rmsds: list, dmaes: list) -> None:
    """Print the RMSE of the energy errors and the mean and std of the RMSDs and D-MAEs."""
    delEs, rmsds, dmaes = np.array(delEs), np.array(rmsds), np.array(dmaes)
    print(f"{prefix}RMSE: ", np.round(np.sqrt(np.mean(delEs**2)), 3).item())
    print(f"{prefix}RMSD: ", np.round(np.mean(rmsds), 3).item(), np.round(np.std(rmsds), 3).item())
    print(f"{prefix}D-MAE: ", np.round(np.mean(dmaes), 3).item(), np.round(np.std(dmaes), 3).item())


def main() -> None:
    args = parse_args()
    for folder in (args.gt_folder, args.gen_folder):
        if not os.path.isdir(folder):
            print(f"Error: {folder} is not a valid folder.")
            sys.exit()

    info = QM_KWDS[args.datatype]
    folds = range(args.folds) if args.folds else [0]
    delEs = {fold: [] for fold in folds}
    rmsds = {fold: [] for fold in folds}
    dmaes = {fold: [] for fold in folds}
    errs = {fold: [] for fold in folds}

    for i, filename in enumerate(os.listdir(args.gt_folder)):
        if not filename.endswith(".xyz"):
            continue
        print(i, filename)
        gt_filepath = os.path.join(args.gt_folder, filename)
        n_atoms_gt = atom_count(gt_filepath)

        for fold in folds:
            gen_filepath = os.path.join(
                args.gen_folder, generated_name(args.pattern, filename, fold)
            )
            if args.folds:
                print(gen_filepath)
            assert n_atoms_gt == atom_count(gen_filepath), "Two files are different!!!"

            try:
                delE = calc_deltaE(
                    gt_filepath, gen_filepath, f=False,
                    xc=info[0], basis=info[1], d3=info[2],
                )
            except Exception as e:
                errs[fold].append(filename)
                print(f"{filename} has an error {e}")
                continue
            rmsd, dmae = rmsd_n_dmae(gt_filepath, gen_filepath)

            delEs[fold].append(abs(delE).item())
            rmsds[fold].append(rmsd.item())
            dmaes[fold].append(dmae.item())

    if args.folds:
        for fold in folds:
            summarise(f"fold_{fold}_", delEs[fold], rmsds[fold], dmaes[fold])
            print("\n")
        print("median RMSD: ", [np.median(rmsds[f]).item() for f in folds])
        print("median DMAE: ", [np.median(dmaes[f]).item() for f in folds])
        print("median delE: ", [np.median(delEs[f]).item() for f in folds])
        return

    print("\n")
    print(args.gen_folder)
    print("delta Es: ", delEs[0])
    print("\n")
    print("rmsds: ", rmsds[0])
    print("\n")
    print("dmaes: ", dmaes[0])
    print("\n")
    print("erros files: ", errs[0])
    summarise("", delEs[0], rmsds[0], dmaes[0])


if __name__ == "__main__":
    main()
