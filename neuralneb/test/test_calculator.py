from ase.io import read

from neuralneb.painn import PaiNN
from neuralneb.utils import MLCalculator


def test_can_predict():
    model = PaiNN(1, 10, 5)
    atoms = read('data/test_reaction/p.xyz')
    atoms.calc = MLCalculator(model)
    atoms.get_potential_energy()
