from typing import List
import numpy as np

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import (
    BruteForceOrderMatcher,
    GeneticOrderMatcher,
    HungarianOrderMatcher,
    KabschMatcher,
)
from pymatgen.io.xyz import XYZ

from torch import Tensor


def xh2pmg(xh):
    mol = Molecule(
        species=xh[:, -1].long().cpu().numpy(),
        coords=xh[:, :3].cpu().numpy(),
    )
    return mol


def xyz2pmg(xyzfile):
    xyz_converter = XYZ(mol=None)
    mol = xyz_converter.from_file(xyzfile).molecule
    return mol


def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
    _, count = np.unique(mol1.atomic_numbers, return_counts=True)
    if same_order:
        bfm = KabschMatcher(mol1)
        prime, rmsd = bfm.fit(mol2)
        return prime, rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        prime, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        rmsds = [np.inf] # threshold #changed
        prime = [np.zeros((1, 3))] #dummy
        for pair in pairs:
            # rmsd = min(rmsd, pair[-1]) 
            rmsds.append(pair[-1]) #changed
            prime.append(pair[0]) #changed
        idx = np.argmin(rmsds) #changed
        rmsd = rmsds[idx] #changed
        prime = prime[idx] #changed
        if not len(pairs):
            bfm = HungarianOrderMatcher(mol1)
            prime, rmsd = bfm.fit(mol2)
    # print("original rmsd: ", rmsd)
    # print("used method: ", method)
    return prime, rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality=False,
    threshold=0.5,
    same_order=False,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    prime, rmsd = rmsd_core(mol1, mol2, threshold)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        prime, rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold)
        rmsd = min(rmsd, rmsd_reflect)
    return prime, rmsd*(3**0.5) #changed


def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold=0.5,
):
    rmsds = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms
        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        try:
            _, rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=False, threshold=threshold)
        except:
            rmsd = 5.0
        rmsds.append(rmsd) #changed
        # rmsds.append(min(rmsd, 1.0)) #changed
        start_ind = end_ind
    return rmsds
