"""Convert a dataset split pickle into the flat .xyz layout TS-GAN reads.

All reactions land in one output directory: the TS as mol_<i>.xyz, the reactant as
mol_<i>_rev.xyz and the product as mol_<i>_for.xyz, all using the reactant atom order.
"""

import pickle
import os
import argparse

ATOM_MAP = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F"
}

# -------- ARGUMENTS --------
parser = argparse.ArgumentParser(description="Convert PKL dataset to XYZ files")
parser.add_argument("--input_pkl", type=str, required=True, help="Path to input .pkl file")
parser.add_argument("--output_dir", type=str, default="xyz_data", help="Output directory")

args = parser.parse_args()

input_pkl = args.input_pkl
output_dir = args.output_dir

os.makedirs(output_dir, exist_ok=True)


# -------- WRITE XYZ --------
def write_xyz(filename, atomic_numbers, coords):
    """Write one .xyz frame, mapping atomic numbers to symbols via ATOM_MAP (H, C, N, O, F only)."""
    with open(filename, "w") as f:
        n = len(atomic_numbers)
        f.write(f"{n}\n")
        f.write("Generated from pickle\n")
        
        for z, (x, y, zc) in zip(atomic_numbers, coords):
            symbol = ATOM_MAP[int(z)]
            f.write(f"{symbol} {x:.6f} {y:.6f} {zc:.6f}\n")


# -------- LOAD DATA --------
with open(input_pkl, "rb") as f:
    data = pickle.load(f)


# -------- CONVERT --------
for i, sample in enumerate(data):

    react = sample["pos_0"].cpu().numpy()
    ts    = sample["pos_1"].cpu().numpy()
    prod  = sample["pos_2"].cpu().numpy()

    atoms = sample["charge_0"].cpu().numpy().flatten()

    base = f"mol_{i}"

    # TS
    write_xyz(os.path.join(output_dir, f"{base}.xyz"), atoms, ts)

    # Reactant
    write_xyz(os.path.join(output_dir, f"{base}_rev.xyz"), atoms, react)

    # Product
    write_xyz(os.path.join(output_dir, f"{base}_for.xyz"), atoms, prod)


print(f"Converted {len(data)} samples to {output_dir}/")