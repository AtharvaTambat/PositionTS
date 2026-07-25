import pickle
import copy
import numpy as np
import torch
import sys

from toxify import create_permutation_matrix

pkl_path = sys.argv[1]
out_path = sys.argv[2]

# -------- LOAD DATA --------
with open(pkl_path, "rb") as f:
    data = pickle.load(f)

data_new = []
# -------- CONVERT --------
print("Creating permuted data ...", flush=True)
for i, sample in enumerate(data):
    print(f"Sample {i}", flush=True)
    sample_new = copy.deepcopy(sample)
    size = sample["size_2"]

    p_pos  = sample["pos_2"]
    p_atoms = sample["charge_2"]
    p_ohe = sample["one_hot_2"]
    pos_dtype = p_pos.dtype
    ohe_dtype = p_ohe.dtype
    atom_dtpe = p_atoms.dtype
    device = p_pos.device

    pms = create_permutation_matrix(size)
    sample_new["pos_2"] = (pms @ p_pos).to(pos_dtype).to(device)
    sample_new["one_hot_2"] = (pms.float() @ p_ohe.float()).to(ohe_dtype).to(device)
    sample_new["charge_2"] = (pms.float() @ p_atoms.float()).to(atom_dtpe).to(device)

    data_new.append(sample_new)

# Saving the list
with open(out_path, 'wb') as f:
    pickle.dump(data_new, f)

print("Done!!!")

    

    