import pytest
from neuralneb.painn import PaiNN


def test_painn_can_instantiate():
    PaiNN(3, 256, 5)
