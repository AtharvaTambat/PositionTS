"""Single-reaction NEB search with a trained PaiNN potential; run_all.sh submits one job per reaction.

Product and reactant .xyz paths are positional, product first. Unlike neb_new.py there is no
climbing image, and the saved TS is the image of minimum fmax rather than of maximum energy.
"""

from argparse import ArgumentParser
import sys
import os
sys.path.append("../neuralneb")

import matplotlib.pyplot as plt
import torch
import numpy as np

from ase.io import read, write
from ase.mep import NEB, NEBTools   # updated import
from ase.optimize import BFGS
from neuralneb import painn, utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):
    # -------------------------------
    # LOAD MODEL
    # -------------------------------
    statedict = torch.load(args.model, map_location=DEVICE)
    model = painn.PaiNN(3, 256, 5)
    model.load_state_dict(statedict)
    model.eval()

    # -------------------------------
    # LOAD STRUCTURES
    # -------------------------------
    product = read(args.product)
    reactant = read(args.reactant)

    assert str(product.symbols) == str(reactant.symbols), \
        "product and reactant must have same formula."

    # -------------------------------
    # INIT IMAGES
    # -------------------------------
    atom_configs = [reactant.copy() for _ in range(10)] + [product]

    # IMPORTANT: separate calculator per image (NEB requirement)
    for atom_config in atom_configs:
        atom_config.calc = utils.MLCalculator(model)

    # -------------------------------
    # OPTIMIZATION (UNCHANGED)
    # -------------------------------
    BFGS(atom_configs[0]).run(fmax=0.05, steps=100)
    BFGS(atom_configs[-1]).run(fmax=0.05, steps=100)

    neb = NEB(atom_configs)
    neb.interpolate(method="idpp")

    optimizer = BFGS(neb)
    optimizer.run(fmax=0.05, steps=20)

    # -------------------------------
    # Save the predicted TS: the image of minimum fmax, not of maximum energy
    # -------------------------------
    forces = [img.get_forces() for img in atom_configs]
    fmax_list = [np.max(np.abs(f)) for f in forces]

    best_idx = int(np.argmin(fmax_list))
    best_ts = atom_configs[best_idx]

    os.makedirs(args.output_dir, exist_ok=True)
    fname = os.path.basename(args.reactant)
    save_path = os.path.join(args.output_dir, fname)

    write(save_path, best_ts)

    print(f"Saved TS: {save_path} | fmax={fmax_list[best_idx]:.6f}", flush=True)

    # -------------------------------
    # ORIGINAL PLOTTING (UNCHANGED)
    # -------------------------------
    nebtools = NEBTools(atom_configs)
    fit = nebtools.get_fit()

    energies = fit.fit_energies.tolist()
    path = fit.fit_path.tolist()

    # mep_fig(path, energies)
    # plt.show()
    # write("/tmp/rxn.gif", images=atom_configs, format="gif")
    # plt.show()


def mep_fig(path, energy):
    """Minimum-energy-path figure, titled with max(energy) in eV. Not used by this script."""
    fig, ax = plt.subplots()
    ax.plot(path, energy, label="MEP")
    ax.grid()
    ax.set_title(f"Barrier height: {str(max(energy))[:5]} eV")
    ax.set_xlabel("Reaction Coordinate [AA]")
    ax.set_ylabel("Energy [eV]")
    ax.legend()
    return fig


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("product")
    parser.add_argument("reactant")

    parser.add_argument("--model", nargs="?", default="data/painn.sd")

    # New argument (only addition)
    parser.add_argument("--output_dir", required=True)

    args = parser.parse_args()

    main(args)