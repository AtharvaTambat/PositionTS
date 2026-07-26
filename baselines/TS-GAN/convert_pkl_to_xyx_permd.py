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
os.makedirs(os.path.join(output_dir, "reactant"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "product"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "ts"), exist_ok=True)

# -------- WRITE XYZ --------
def write_xyz(filename, atomic_numbers, coords):
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

    atoms_r = sample["charge_0"].cpu().numpy().flatten()
    atoms_ts = sample["charge_1"].cpu().numpy().flatten()
    atoms_p = sample["charge_2"].cpu().numpy().flatten()

    base = f"entry_{i}"

    # TS
    write_xyz(os.path.join(output_dir, "ts", f"{base}.xyz"), atoms_ts, ts)

    # Reactant
    write_xyz(os.path.join(output_dir, "reactant", f"{base}.xyz"), atoms_r, react)

    # Product
    write_xyz(os.path.join(output_dir, "product", f"{base}.xyz"), atoms_p, prod)


print(f"✅ Converted {len(data)} samples to {output_dir}/")