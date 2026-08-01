"""Run a nudged-elastic-band search for one reactant/product pair using a trained PaiNN potential."""
from argparse import ArgumentParser
import sys
sys.path.append("../neuralneb")
import matplotlib.pyplot as plt
import torch

from ase.io import read, write
# from ase.neb import NEB, NEBOptimizer, NEBTools
from ase.neb import NEB, NEBTools
from ase.optimize import BFGS
# from ase.optimize.bfgs import BFGS
from ase.optimize import BFGS
from neuralneb import painn, utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(args):  # pylint: disable=redefined-outer-name
    statedict = torch.load(args.model)
    model = painn.PaiNN(3, 256, 5)
    model.load_state_dict(statedict)
    model.eval()

    product = read(args.product)
    reactant = read(args.reactant)
    assert str(product.symbols) == str(reactant.symbols), "product and reactant must have same formula. Product: {product.symbols}, Reactant: {reactant.symbols}"
    atom_configs = [reactant.copy() for _ in range(10)] + [product]

    for atom_config in atom_configs:
        atom_config.calc = utils.MLCalculator(model)

    BFGS(atom_configs[0]).run(fmax=0.05, steps=1000)
    BFGS(atom_configs[-1]).run(fmax=0.05, steps=1000)

    neb = NEB(atom_configs)
    neb.interpolate(method="idpp")
    # relax_neb = NEBOptimizer(neb)
    # relax_neb.run()
    optimizer = BFGS(neb)
    optimizer.run(fmax=0.05,  steps=200)

    nebtools = NEBTools(atom_configs)
    fit = nebtools.get_fit()

    energies = fit.fit_energies.tolist()
    path = fit.fit_path.tolist()

    # mep_fig(path, energies)
    # plt.show()
    # write("/tmp/rxn.gif", images=atom_configs, format="gif")
    # plt.show()


def mep_fig(path, energy):
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
    parser.add_argument("product", nargs="?", default="data/test_reaction/p.xyz")
    parser.add_argument("reactant", nargs="?", default="data/test_reaction/r.xyz")
    parser.add_argument("--model", nargs="?", default="data/painn.sd")
    args = parser.parse_args()

    main(args)
