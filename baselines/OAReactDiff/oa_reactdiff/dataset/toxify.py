"""Atom-order permutation matrices for the atom-order robustness experiment.
Applied by ProcessedTS1x.permute_fragments to relabel the atoms of selected fragments.
"""
import numpy as np
import torch

def permutated_idxs(nodes):
    """Atom indices of a batch shuffled within each molecule's contiguous block.
    Reseeds numpy with 42 on every call, so a given block layout always gets the same
    permutation (and the global numpy RNG is reset as a side effect).
    """
    np.random.seed(42)
    perm_indices = []
    start = 0
    for i, num in enumerate(nodes):
        perms = list(range(start, start + num))
        np.random.shuffle(perms)
        perm_indices.extend(perms)
        start += num
    
    return perm_indices

def create_permutation_matrix(nodes):
    """Square permutation matrix P over the batch's atoms, such that P @ x reorders
    atoms within each molecule according to permutated_idxs(nodes)."""
    permutation_indices = permutated_idxs(nodes)
    n = len(permutation_indices)
    # Initialize an n x n matrix of zeros
    P = torch.zeros((n, n), device=nodes.device)
    # Place a 1 at the specified (row, column) locations
    for i, j in enumerate(permutation_indices):
        P[i, j] = 1
    P = torch.tensor(P)#.double()
    return P
