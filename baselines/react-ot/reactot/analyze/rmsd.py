from typing import List
import numpy as np
import math

from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher, GeneticOrderMatcher, HungarianOrderMatcher, KabschMatcher
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


# def rmsd_core(mol1, mol2, threshold=0.5, same_order=False):
#     _, count = np.unique(mol1.atomic_numbers, return_counts=True)
#     if same_order:
#         bfm = KabschMatcher(mol1)
#         _, rmsd = bfm.fit(mol2)
#         return rmsd
#     total_permutations = 1
#     for c in count:
#         total_permutations *= np.math.factorial(c)  # type: ignore
#     if total_permutations < 1e4:
#         bfm = BruteForceOrderMatcher(mol1)
#         _, rmsd = bfm.fit(mol2)
#     else:
#         bfm = GeneticOrderMatcher(mol1, threshold=threshold)
#         pairs = bfm.fit(mol2)
#         rmsd = threshold
#         for pair in pairs:
#             rmsd = min(rmsd, pair[-1])
#         if not len(pairs):
#             bfm = HungarianOrderMatcher(mol1)
#             _, rmsd = bfm.fit(mol2)
#     return rmsd

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
            _, rmsd = bfm.fit(mol2)
    return rmsd


def pymatgen_rmsd(
    mol1,
    mol2,
    ignore_chirality: bool = False,
    threshold: float = 0.5,
    same_order: bool = True,
):
    if isinstance(mol1, str):
        mol1 = xyz2pmg(mol1)
    if isinstance(mol2, str):
        mol2 = xyz2pmg(mol2)
    rmsd = rmsd_core(mol1, mol2, threshold, same_order=same_order)
    if ignore_chirality:
        coords = mol2.cart_coords
        coords[:, -1] = -coords[:, -1]
        mol2_reflect = Molecule(
            species=mol2.species,
            coords=coords,
        )
        rmsd_reflect = rmsd_core(
            mol1, mol2_reflect, threshold, same_order=same_order)
        rmsd = min(rmsd, rmsd_reflect)
    return rmsd


def batch_rmsd(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold: float = 0.5,
    same_order: bool = False,
) -> List[float]:
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
            rmsd = pymatgen_rmsd(
                mol1,
                mol2,
                # ============================================================================
                # FLAW (this work, paper Sec. 5.1(II))
                # Two issues on this call.
                # 1. ignore_chirality=True also matches the z-reflected geometry, so an
                #    enantiomer of the true transition state scores as correct.
                # 2. pymatgen_rmsd omits the sqrt(3) factor that undoes pymatgen's extra
                #    averaging over the three Cartesian components, so values are smaller
                #    by sqrt(3) than the definition in Sec. 5. analyze/rmsd_corrected.py
                #    applies the factor but is imported nowhere.
                # ============================================================================
                ignore_chirality=True,
                threshold=threshold,
                same_order=same_order,
            )
        except:
            # ============================================================================
            # FLAW (this work, paper Sec. 5.1(II))
            # A failed match is recorded as 1 rather than skipped, so matcher failures
            # enter the reported statistics as if they were measurements.
            # ============================================================================
            rmsd = 1
        # ============================================================================
        # FLAW (this work, paper Sec. 5.1(II))
        # RMSD is capped at 1.0 A. Large errors are truncated, which compresses the
        # reported spread and flatters poor predictions.
        # ============================================================================
        rmsds.append(min(rmsd, 1.0))
        start_ind = end_ind
    return rmsds


def batch_rmsd_n_dmae(
    fragments_nodes: List[Tensor],
    out_samples: List[Tensor],
    xh: List[Tensor],
    idx: int = 1,
    threshold: float = 0.5,
    same_order: bool = False,
) -> List[float]:
    rmsds = []
    dmaes = []
    out_samples_use = out_samples[idx]
    xh_use = xh[idx]
    nodes = fragments_nodes[idx].long().cpu().numpy()
    start_ind, end_ind = 0, 0
    for jj, natoms in enumerate(nodes):
        end_ind += natoms
        mol1 = xh2pmg(out_samples_use[start_ind:end_ind])
        mol2 = xh2pmg(xh_use[start_ind:end_ind])
        prime, rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=False, threshold=threshold)
        # try:
        #     _, rmsd = pymatgen_rmsd(mol1, mol2, ignore_chirality=False, threshold=threshold)
        # except Exception as e:
        #     print("error :", e)
        #     rmsd = 5.0
        dmae = dmae_core(mol1, prime)
        # ============================================================================
        # FLAW (this work, paper Sec. 5.1(II))
        # Corrected variant: this appends the raw rmsd. The capped form is retained
        # two lines below but commented out, unlike batch_rmsd and batch_rmsd_sb
        # above, which still apply the 1.0 A cap.
        # ============================================================================
        rmsds.append(rmsd) #changed
        dmaes.append(dmae)
        # rmsds.append(min(rmsd, 1.0)) #changed
        start_ind = end_ind
    return rmsds, dmaes


def batch_rmsd_sb(
    fragments_node: Tensor,
    pred_xh: Tensor,
    target_xh: Tensor,
    threshold: float = 0.5,
    same_order: bool = True,
) -> List[float]:

    rmsds = []

    end_ind = np.cumsum(fragments_node.long().cpu().numpy())
    start_ind = np.concatenate([np.int64(np.zeros(1)), end_ind[:-1]])

    for start, end in zip(start_ind, end_ind):
        mol1 = xh2pmg(pred_xh[start : end])
        mol2 = xh2pmg(target_xh[start : end])
        rmsd = pymatgen_rmsd(
            mol1,
            mol2,
            # ============================================================================
            # FLAW (this work, paper Sec. 5.1(II))
            # ignore_chirality=True also matches the z-reflected geometry, so an
            # enantiomer of the true transition state scores as correct.
            # ============================================================================
            ignore_chirality=True,
            threshold=threshold,
            same_order=same_order,
        )
        # ============================================================================
        # FLAW (this work, paper Sec. 5.1(II))
        # RMSD is capped at 1.0 A, truncating large errors.
        # ============================================================================
        rmsds.append(min(rmsd, 1.0))
    return rmsds
