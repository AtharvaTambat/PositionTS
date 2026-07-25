import numpy as np
import sys
import os
import pickle
import torch
import D2C
import atom_data

reac_folder = sys.argv[1]
prod_folder = sys.argv[2]
ts_D_file = sys.argv[3]

out_folder = sys.argv[-1]
os.makedirs(out_folder, exist_ok=True)

def get_D(X, atoms):
    arr = []
    a_d = atom_data.symbol_dict(sys.argv[0])
    for i in range (len(atoms)):
        arr.append([])
        for j in range (len(atoms)):
            if i != j:
                r = (a_d[atoms[i]] * a_d[atoms[j]]) / X[i][j]
            else:
                r = 0.0
            arr[-1].append(r)

    # take average to make symmetric
    for i in range (len(atoms)):
        for j in range (i + 1, len(atoms)):
            arr[i][j] = (arr[i][j] + arr[j][i])/2

    for i in range (len(atoms)):
        for j in range (i + 1):
            arr[i][j] = arr[j][i]

    return np.array(arr)

def get_atoms(a, b):
    g = open(a)
    lines = g.readlines()[2:]
    g.close()
    at = []
    cords = []
    for line in lines:
        k = line.strip().split()
        if len(k) < 4:
            continue
        at.append(k[0])
        cords.append(list(map(float, k[1:])))

    g = open(b)
    lines = g.readlines()[2:]
    g.close()
    at2 = []
    cords2 = []
    for line in lines:
        k = line.strip().split()
        if len(k) < 4:
            continue
        at2.append(k[0])
        cords2.append(list(map(float, k[1:])))

    # Take mean of reactant and product
    #cords = np.mean([np.array(cords), np.array(cords2)], axis = 0)
    return at, np.array(cords)

print("Loading the pkl file ...", flush=True)
# Open the file in binary mode
with open(ts_D_file, 'rb') as file:
    data = pickle.load(file)
print(len(data))
print(len(data[0]))
print("pkl file loaded.", flush=True)
print("Processing coords generation ...", flush=True)
print("="*10, flush=True)

for fname in os.listdir(reac_folder):
    if not fname.endswith(".xyz"):
        continue
    
    r_file = os.path.join(reac_folder, fname)
    p_file = os.path.join(prod_folder, fname)

    atoms, cords = get_atoms(r_file, p_file)

    idx = int(fname.split("_")[-1].replace(".xyz", ""))
    D_matrix = data[0][idx]
    D_matrix = D_matrix.numpy()
    print(f"Processing {fname}", flush=True)
    new_cords = D2C.job(D_matrix, cords, atoms)

    out_file = os.path.join(out_folder, fname.replace('.xyz', '_ts.xyz'))

    with open(out_file, 'w') as g:
        g.write(str(len(atoms)) + "\nGenerated TS\n")
        for j in range(len(atoms)):
            g.write(atoms[j] + ' ' + ' '.join(map(str, new_cords[j])) + '\n')

print("Done!!!", flush=True)