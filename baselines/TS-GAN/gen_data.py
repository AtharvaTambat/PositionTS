"""Build the Coulomb-matrix .npy training set from a directory of .xyz files."""
import numpy as np
import os
import glob
import argparse
import xyz_to_clmb as clb

# -------- ARGUMENTS --------
parser = argparse.ArgumentParser(description="Generate Coulomb matrix dataset from xyz files")
parser.add_argument("--xyz_dir", type=str, required=True, help="Directory containing xyz files")
parser.add_argument("--output_path", type=str, default="data.npy", help="Output .npy file")

args = parser.parse_args()

xyz_dir = args.xyz_dir
output_path = args.output_path


ma = 100

pr, re, ts, nam = [], [], [], []

files_set = set(os.listdir(xyz_dir))  # faster lookup

# -------- MAIN LOOP --------
for f in glob.glob(os.path.join(xyz_dir, "*_rev.xyz")):

    fname = os.path.basename(f)

    # Extract molecule id more safely
    base = fname.replace("_rev.xyz", "")   # e.g. mol_0

    f_for = os.path.join(xyz_dir, base + "_for.xyz")
    f_ts  = os.path.join(xyz_dir, base + ".xyz")

    if not (os.path.exists(f_for) and os.path.exists(f_ts)):
        print(f"Missing pair for {base}, skipping")
        continue

    try:
        # Reactant
        rev = np.array(clb.job(f))

        # Product
        prod = np.array(clb.job(f_for))

        # TS
        tsl = np.array(clb.job(f_ts))

        if rev.shape == (ma, ma) and prod.shape == (ma, ma) and tsl.shape == (ma, ma):
            re.append(rev)
            pr.append(prod)
            ts.append(tsl)
            nam.append(base)

    except Exception as e:
        print(f"Error processing {base}: {e}")
        continue


# -------- SAVE --------
data = {
    'products': np.array(pr),
    'reactants': np.array(re),
    'ts': np.array(ts),
    'names': np.array(nam)
}

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

np.save(output_path, data)

print(f"Saved {len(pr)} samples to {output_path}")