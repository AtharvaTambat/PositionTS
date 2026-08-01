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
        _, rmsd = bfm.fit(mol2)
        return rmsd
    total_permutations = 1
    for c in count:
        total_permutations *= np.math.factorial(c)  # type: ignore
    if total_permutations < 1e4:
        bfm = BruteForceOrderMatcher(mol1)
        _, rmsd = bfm.fit(mol2)
    else:
        bfm = GeneticOrderMatcher(mol1, threshold=threshold)
        pairs = bfm.fit(mol2)
        # ============================================================================
        # FLAW (this work, paper Sec. 5.1(II))
        # Implicit cap: rmsd is seeded with `threshold` and only ever lowered by the
        # min() below, so when every GeneticOrderMatcher pair exceeds threshold the
        # reported value is threshold (0.5 A by default), not the true RMSD. This is
        # separate from, and additional to, the explicit 1.0 A cap further down.
        # (An empty `pairs` is handled by the HungarianOrderMatcher branch instead.)
        # ============================================================================
        rmsd = threshold
        for pair in pairs:
            rmsd = min(rmsd, pair[-1])
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
    rmsd = rmsd_core(mol1, mol2, threshold)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(mol1, mol2_reflect, threshold)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


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
            # ============================================================================
            # FLAW (this work, paper Sec. 5.1(II))
            # Two issues on the call below.
            # 1. ignore_chirality=True also matches the z-reflected geometry, so an
            #    enantiomer of the true transition state scores as correct.
            # 2. pymatgen_rmsd omits the sqrt(3) factor that undoes pymatgen's extra
            #    averaging over the three Cartesian components, so values here are
            #    smaller by sqrt(3) than the definition in Sec. 5 of the paper.
            #    analyze/rmsd_corrected.py applies the factor but is imported nowhere.
            # ============================================================================
            rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=True, threshold=threshold)
        except:
            # ============================================================================
            # FLAW (this work, paper Sec. 5.1(II))
            # A failed match is recorded as 1.0 A rather than skipped, so matcher
            # failures enter the reported statistics as if they were measurements.
            # ============================================================================
            rmsd = 1.0
        # ============================================================================
        # FLAW (this work, paper Sec. 5.1(II))
        # RMSD is capped at 1.0 A. Large errors are truncated, which compresses
        # the reported spread and flatters poor predictions.
        # ============================================================================
        rmsds.append(min(rmsd, 1.0))
        start_ind = end_ind
    return rmsds
