import os
import sys
import numpy as np
# import torch

from toxify import permutated_idxs

input_folder = sys.argv[1]
output_folder = sys.argv[2]

def shuffle_coords(xyz_lines):
    xyz_lines_mod = xyz_lines[-1] + "\n"
    xyz_lines[-1] = xyz_lines_mod
    n_atoms = int(xyz_lines[0].strip())
    # print(n_atoms)
    perm_idx = np.array(permutated_idxs(n_atoms))
    print(perm_idx)
    SymX = xyz_lines[2:]
    SymX_permd = np.array(SymX)[perm_idx]
    SymX_permd = SymX_permd.tolist()
    output_lines = xyz_lines[:1] + ["shuffled TS\n"] + SymX_permd
    return output_lines

def write_xyz(output_path, xyz_lines):
    """
    Write XYZ content to a file.

    Parameters
    ----------
    output_path : str
        Path to the output .xyz file.
    xyz_lines : list[str] or str
        XYZ content as a list of lines or a single string.
    """
    with open(output_path, "w") as f:
        if isinstance(xyz_lines, list):
            f.write("".join(xyz_lines))
        else:
            f.write(str(xyz_lines))


os.makedirs(output_folder, exist_ok=True)
for filename in os.listdir(input_folder):
    if not filename.endswith(".xyz"):
        continue
    input_filepath = os.path.join(input_folder, filename)
    try:
        with open(input_filepath, 'r') as f:
            input_lines = f.readlines()
    except FileNotFoundError:
        msg = f"Error: file {input_filepath} not found"
        print(msg)
        continue

    shuffled_lines = shuffle_coords(input_lines)
    output_path = os.path.join(output_folder, filename)
    write_xyz(output_path, shuffled_lines)
        