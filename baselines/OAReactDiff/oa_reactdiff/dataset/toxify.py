import numpy as np
import torch

def permutated_idxs(nodes):
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
    """
    Creates a permutation matrix from a list of permutation indices.

    Args:
        permutation_indices: A list/array where the value at index i
                             indicates the new position (column j) for
                             the 1 in row i. E.g., [1, 2, 0] means
                             row 0 has 1 in column 1, row 1 in column 2,
                             and row 2 in column 0.
    Returns:
        A NumPy permutation matrix (2D array).
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
