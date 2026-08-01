"""Maximum and average molecular weight and heavy-atom count of the reactants across a dataset's three splits."""
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
import sys
import pandas as pd
import os
import numpy as np

data_folder = sys.argv[1]

def calculate_molecular_weight(mol):
    if mol is not None:
        return Descriptors.MolWt(mol)
    else:
        return None

def calculate_heavy_atoms(mol):
    if mol is not None:
        return rdMolDescriptors.CalcNumHeavyAtoms(mol)
    else:
        return None

def prepare_mol(xyz, chg=0):
    """Build an RDKit molecule from an xyz block, perceiving connectivity and bond orders.

    A bond-order failure is printed and swallowed, so the molecule returned may carry
    connectivity only.
    """
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    try:
        rdDetermineBonds.DetermineBondOrders(mol, charge=chg)
    except Exception as e:
        print(f"Error determining bond orders: {e}")
    sanitize_status = Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL,
        catchErrors=True,
    )
    # print('Charge', chg)
    return mol

failed = []
mws = [] 
heavy_atoms = []
for splits in ["train", "test", "validation"]:
    split_folder = os.path.join(data_folder, f"{splits}_split", "reactants")
    for files in os.listdir(split_folder):
        if not files.endswith(".xyz"):
            continue
        input_filepath = os.path.join(split_folder, files)
        try:
            with open(input_filepath, 'r') as f:
                input_lines = f.readlines()
        except FileNotFoundError:
            msg = f"Error: file {input_filepath} not found"
            print(msg)
            continue
            
        try:
            mol = prepare_mol("".join(input_lines))
            mws.append(calculate_molecular_weight(mol))
            heavy_atoms.append(calculate_heavy_atoms(mol))
        except:
            failed.append(input_filepath)

mws = np.array(mws)
heavy_atoms = np.array(heavy_atoms)
# print the results
# print("data file: ", data_file)
print("Maximum molecular weight: ", mws.max())
print("Average molecular weight: ", mws.mean())
print("Maximum number of heavy atoms: ", heavy_atoms.max())
print("Average number of heavy atoms: ", heavy_atoms.mean())