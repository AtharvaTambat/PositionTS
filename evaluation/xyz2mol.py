import os
import sys

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds


def prepare_mol(xyz, chg=0):
    mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(mol)
    rdDetermineBonds.DetermineBondOrders(mol, charge=chg)
    # print(f"Error determining bond orders: {e}")
    sanitize_status = Chem.SanitizeMol(
        mol,
        sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL,
        catchErrors=True,
    )
    # print('Charge', chg)
    return mol
    
xyz_folder = sys.argv[1]
cp_folder = sys.argv[2]
failed = []
failed_fp = []
for i, filename in enumerate(os.listdir(xyz_folder)):
    if not filename.endswith(".xyz"):
        continue  

    input_filepath = os.path.join(xyz_folder, filename)
    print(i)
    try:
        with open(input_filepath, 'r') as f:
            input_lines = f.readlines()
    except FileNotFoundError:
        msg = f"Error: file {input_filepath} not found"
        print(msg)
        continue
        
    try:
        mol = prepare_mol("".join(input_lines))
    except:
        failed.append(filename)
        failed_fp.append(input_filepath)

        # print(failed_fp)
        # if len(failed_fp)>2:
        #     exit()


print(failed)
print("\n")
print(len(failed))

for fps in failed_fp:
    print(f"cp {fps} {cp_folder}")