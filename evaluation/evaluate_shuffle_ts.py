"""Compare generated transition states against the reference and its atom-order-permuted copy.

Writes the per-reaction atom count and the (original, shuffled) RMSD and D-MAE pairs
to <datatype>.json.
"""
from rmsd import rmsd_n_dmae
import time
import os
import sys
import numpy as np
import json

gen_folder = sys.argv[1]
datatype = sys.argv[2]

gt_folder = f"../baselines/ts_gen/data/{datatype}/xyz_files/test_split/transition_states"
gt_shuf_folder = f"../baselines/ts_gen/data/{datatype}/xyz_files/test_split/transition_states_shuffd"
qm_kwds = {"transition1x": ["wb97x", "631g*", False],
           "isomer": ["wb97x", "def2-TZVP", True],
           "rgd1": ["b3lyp", "def2-TZVP", True],
           "rgdx": ["b3lyp", "def2-TZVP", True],
           "rgdy": ["b3lyp", "def2-TZVP", True],
           "rgdx10": ["b3lyp", "def2-TZVP", True]}

if not os.path.isdir(gt_folder):
    print(f"Error: {gt_folder} is not a valid folder.")
    sys.exit()

if not os.path.isdir(gt_shuf_folder):
    print(f"Error: {gt_shuf_folder} is not a valid folder.")
    sys.exit()

if not os.path.isdir(gen_folder):
    print(f"Error: {gen_folder} is not a valid folder.")
    sys.exit()

info = qm_kwds[datatype]

rmsds_orig = []
dmaes_orig = []

rmsds_shuf = []
dmaes_shuf = []

difference = {}
errs = []
for i, filename in enumerate(os.listdir(gt_folder)):
    if not filename.endswith(".xyz"):
        continue  
    print(i, filename)
    gt_filepath = os.path.join(gt_folder, filename)
    shuf_filepath = os.path.join(gt_shuf_folder, filename)
    with open(gt_filepath, 'r') as f:
        gt_lines = f.readlines()
    n_atoms_gt = int(gt_lines[0].strip())
    
    filen = filename.split(".")[0]
    gen_filepath = os.path.join(gen_folder, f"{filen}_ts.xyz")
    # filen = filename.split("_")[-1]
    # gen_filepath = os.path.join(gen_folder, f"tsgen_idx_{filen}")
    with open(gen_filepath, 'r') as f:
        gen_lines = f.readlines()
        n_atoms_gen = int(gen_lines[0].strip())

    assert (n_atoms_gt==n_atoms_gen), "Two files are different!!!"

    rmsd_o, dmae_o = rmsd_n_dmae(gt_filepath, gen_filepath)
    rmsds_orig.append(rmsd_o.item())
    dmaes_orig.append(dmae_o.item())

    rmsd_s, dmae_s = rmsd_n_dmae(shuf_filepath, gen_filepath)
    rmsds_shuf.append(rmsd_s.item())
    dmaes_shuf.append(dmae_s.item())

    difference[filename] = [n_atoms_gt, (rmsd_o, rmsd_s), (dmae_o, dmae_s)]
    
json_str = json.dumps(difference, indent=4)
with open(f"{datatype}.json", "w") as f:
    f.write(json_str)
    