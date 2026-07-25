from argparse import ArgumentParser
import os
import sys
import time
import numpy as np

sys.path.append("../neuralneb")

import torch
from ase.io import read, write
from ase.neb import NEB
from ase.optimize import BFGS
from neuralneb import painn, utils

DEVICE = "cpu"  # keep CPU to avoid CUDA issues


# -------------------------------
# LOGGER (prints + writes file)
# -------------------------------
def log(msg, logfile):
    print(msg, flush=True)
    with open(logfile, "a") as f:
        f.write(msg + "\n")


# -------------------------------
# PROCESS ONE REACTION
# -------------------------------
def process_reaction(fname, args, model):
    start_time = time.time()

    reactant_path = os.path.join(args.reactant_dir, fname)
    product_path = os.path.join(args.product_dir, fname)

    reactant = read(reactant_path)
    product = read(product_path)

    atom_configs = [reactant.copy() for _ in range(10)] + [product]

    for atom in atom_configs:
        atom.calc = utils.MLCalculator(model)

    # silent optimization
    BFGS(atom_configs[0], logfile=None).run(fmax=0.05, steps=300)
    BFGS(atom_configs[-1], logfile=None).run(fmax=0.05, steps=300)

    neb = NEB(atom_configs, climb=True)
    neb.interpolate(method="idpp")

    optimizer = BFGS(neb, logfile=None)
    optimizer.run(fmax=0.05, steps=800)

    # Extract energies
    energies = [img.get_potential_energy() for img in atom_configs]

    # Compute fmax
    fmax = max(np.max(np.abs(img.get_forces())) for img in atom_configs)

    # TS = highest energy image
    ts_index = int(np.argmax(energies))
    ts_atoms = atom_configs[ts_index]

    os.makedirs(args.output_dir, exist_ok=True)
    write(os.path.join(args.output_dir, fname), ts_atoms)

    elapsed = time.time() - start_time

    return elapsed, fmax, energies[ts_index]


# -------------------------------
# MAIN
# -------------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    logfile = os.path.join(args.output_dir, "progress.log")

    log("🚀 Script started", logfile)

    # load model ONCE
    statedict = torch.load(args.model, map_location=DEVICE)
    model = painn.PaiNN(3, 256, 5)
    model.load_state_dict(statedict)
    model.eval()

    files = sorted([
        f for f in os.listdir(args.reactant_dir)
        if f.startswith("entry_") and f.endswith(".xyz")
    ])

    total = len(files)

    log(f"Found {total} reactions", logfile)

    if total == 0:
        log("❌ No files found. Exiting.", logfile)
        return

    times = []
    start_global = time.time()

    for i, fname in enumerate(files):
        try:
            log(f"Starting {fname}", logfile)

            elapsed, fmax, energy = process_reaction(fname, args, model)
            times.append(elapsed)

            avg = np.mean(times)
            remaining = total - (i + 1)
            eta = remaining * avg

            percent = 100 * (i + 1) / total

            log(
                f"[{i+1}/{total} | {percent:.1f}%] {fname} | "
                f"fmax={fmax:.4f} | E={energy:.4f} | "
                f"time={elapsed:.2f}s | ETA={eta/60:.2f} min",
                logfile
            )

        except Exception as e:
            log(f"[{i+1}/{total}] {fname} FAILED: {e}", logfile)

    total_time = time.time() - start_global
    log(f"\n✅ ALL DONE in {total_time/60:.2f} minutes", logfile)


# -------------------------------
# ENTRY
# -------------------------------
if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--reactant_dir", type=str, required=True)
    parser.add_argument("--product_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model", nargs="?", default="data/painn.sd")

    args = parser.parse_args()

    main(args)