from geomopt import calc_deltaE
from rmsd import rmsd_n_dmae
import time
import os
import sys
import numpy as np

gt_folder = sys.argv[1]
gen_folder = sys.argv[2]

# datatype = sys.argv[3]
qm_kwds = {"transition1x": ["wb97x", "631g*", False],
           "isomer": ["wb97x", "def2-TZVP", True],
           "rgd1": ["b3lyp", "def2-TZVP", True],
           "rgdx": ["b3lyp", "def2-TZVP", True],
           "rgdy": ["b3lyp", "def2-TZVP", True],
           "rgdx10": ["b3lyp", "def2-TZVP", True]}

if not os.path.isdir(gt_folder):
    print(f"Error: {gt_folder} is not a valid folder.")
    sys.exit()

if not os.path.isdir(gen_folder):
    print(f"Error: {gen_folder} is not a valid folder.")
    sys.exit()

datatype = os.path.basename(os.path.normpath(gen_folder))#.split("_")[0]
info = qm_kwds[datatype]

delEs = []
rmsds = []
dmaes = []
errs = []
for i, filename in enumerate(os.listdir(gt_folder)):
    if not filename.endswith(".xyz"):
        continue  
    print(i, filename)
    gt_filepath = os.path.join(gt_folder, filename)
    with open(gt_filepath, 'r') as f:
        gt_lines = f.readlines()
    n_atoms_gt = int(gt_lines[0].strip())

    # try:
    #     gen_filepath = os.path.join(gen_folder, filename)
    #     with open(gen_filepath, 'r') as f:
    #         gen_lines = f.readlines()
    #     n_atoms_gen = int(gen_lines[0].strip())

    # except:
    #     print(f"No file found: {filename}")
    #     continue

    # gen_filepath = os.path.join(gen_folder, filename)
    filen = filename.split(".")[0]
    gen_filepath = os.path.join(gen_folder, f"{filen}_ts.xyz")
    # filen = filename.split("_")[-1]
    # gen_filepath = os.path.join(gen_folder, f"tsgen_idx_{filen}")
    with open(gen_filepath, 'r') as f:
        gen_lines = f.readlines()
        n_atoms_gen = int(gen_lines[0].strip())

    assert (n_atoms_gt==n_atoms_gen), "Two files are different!!!"

    try:
        delE = calc_deltaE(gt_filepath, gen_filepath, f=False, xc=info[0], basis=info[1], d3=info[2])
    except Exception as e:
        errs.append(filename)
        print(f"{filename} has an error {e}")
        continue
    rmsd, dmae = rmsd_n_dmae(gt_filepath, gen_filepath)

    delEs.append(abs(delE).item())
    rmsds.append(rmsd.item())
    dmaes.append(dmae.item())
    

print("\n")
print(gen_folder)
print("delta Es: ", delEs)
print("\n")
print("rmsds: ", rmsds)
print("\n")
print("dmaes: ", dmaes)
print("\n")
print("erros files: ", errs)

delEs = np.array(delEs)
rmsds = np.array(rmsds)
dmaes = np.array(dmaes)

rmse = np.sqrt(np.mean(delEs**2))
av_rmsd = np.mean(rmsds)
std_rmsd = np.std(rmsds)
av_dmae = np.mean(dmaes)
std_dmae = np.std(dmaes)

print("RMSE: ", np.round(rmse, 3).item())
print("RMSD: ", np.round(av_rmsd, 3).item(), np.round(std_rmsd, 3).item())
print("D-MAE: ", np.round(av_dmae, 3).item(), np.round(std_dmae, 3).item())
    