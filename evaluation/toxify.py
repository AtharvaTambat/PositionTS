"""Fixed atom-order permutations shared by the permuted-copy scripts."""
import numpy as np
import torch

def permutated_idxs(nodes):
    """Shuffled range(nodes); the seed is fixed, so a given size always gives the same order."""
    np.random.seed(42)
    perm_indices = []
    perms = list(range(nodes))
    np.random.shuffle(perms) 
    return perms

def create_permutation_matrix(nodes):
    """Square torch permutation matrix P over `nodes` atoms, so that P @ x reorders rows by permutated_idxs(nodes).

    `nodes` must be a tensor: its device is the one P is built on.
    """
    permutation_indices = permutated_idxs(nodes)
    n = len(permutation_indices)
    # Initialize an n x n matrix of zeros
    P = torch.zeros((n, n), device=nodes.device)
    # Place a 1 at the specified (row, column) locations
    for i, j in enumerate(permutation_indices):
        P[i, j] = 1
    P = torch.tensor(P)#.double()
    return P
