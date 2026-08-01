"""DFT energies, forces and Hessians for transition-state geometries, via PySCF and GPU4PySCF.

Supplies the energy error used by evaluate.py and the imaginary-frequency count used as
the saddle-point test.
"""
import numpy as np

from pyscf import gto, dft
from pyscf.dispersion import dftd3
# from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo
# from pyscf import dftd3

# from molSimplify.Classes.mol3D import mol3D
from pymatgen.core import Molecule
from pymatgen.analysis.molecule_matcher import BruteForceOrderMatcher
from pymatgen.io.xyz import XYZ

# from .rmsd import pymatgen_rmsd

AU2KCALMOL = 627.509608
AU2EV = 27.2114
BOHR = 0.52917721092


def count_negative_eig(x: list):
    """Number of imaginary harmonic frequencies, i.e. entries with a positive imaginary part."""
    count = 0
    for _x in x:
        if _x.imag > 0:
            count += 1
    return count


def _to_numpy(x):
    # GPU4PySCF stores arrays as CuPy on GPU; move them back when needed
    return x.get() if hasattr(x, "get") else x

def compute_efh_gpu(geomfile,f=True,hess=False,return_metrics=False,xc="wb97x",basis="631g*",d3=False,use_gpu=True,auxbasis=None):
    """Density-fitted closed-shell DFT single point, returning the mean-field object, forces and Hessian.

    Forces are in Hartree/Angstrom; with return_metrics the tuple is extended with the
    force RMS in eV/Angstrom and the number of imaginary frequencies (None without hess).
    """
    spin = 0

    mol = gto.M(
        atom=geomfile,
        unit="Ang",
        basis=basis,
        spin=spin,
    )
    mol.build()
    
    mf = dft.RKS(mol) if spin == 0 else dft.UKS(mol)
    mf = mf.density_fit(auxbasis='def2-universal-jkfit')
    mf.xc = xc
    
    if d3:
        mf.disp = "d3zero"   # GPU4PySCF supports D3/D4, but this part is CPU-side

    if use_gpu:
        mf = mf.to_gpu()
        
    mf.conv_tol = 1e-6
    mf.max_cycle = 200
    mf.max_memory = 32000
    mf.kernel()

    force = None
    force_rms = np.nan
    freq_info = None
    hessian = None

    if mf.converged and f:
        g = mf.nuc_grad_method()
        if hasattr(g, "auxbasis_response"):
            g.auxbasis_response = True
        force = -_to_numpy(g.kernel()) / BOHR
        force_rms = np.sqrt(np.mean(force ** 2)) * AU2EV
        print("force rms (ev/A):", force_rms)

    if hess:
        hobj = mf.Hessian()
        if hasattr(hobj, "auxbasis_response"):
            hobj.auxbasis_response = 2
        hessian = _to_numpy(hobj.kernel())

        freq_info = thermo.harmonic_analysis(mf.mol, hessian)
        print("freq:", freq_info["freq_wavenumber"])


    if use_gpu:
        mf = mf.to_cpu()

    if return_metrics:
        nneg = count_negative_eig(freq_info["freq_wavenumber"]) if freq_info is not None else None
        return mf, force, hessian, force_rms, nneg

    return mf, force, hessian

def compute_efh(geomfile, f=True, hess=False, return_metrics=False, xc="wb97x", basis="631g*", d3=False):
    """CPU counterpart of compute_efh_gpu, without density fitting.

    return_metrics only works together with hess, since the frequencies it reports are
    computed inside the Hessian branch.
    """
    spin = 0
    mol = gto.M(
        atom=geomfile,
        unit="Ang",
        basis=basis,
    )
    mol.build()

    mf = dft.RKS(mol) if not spin else dft.UKS(mol)
    mf.xc = xc
    # if d3:
    #     mf = dftd3.dftd3(dft.RKS(mol, xc=xc))
    if d3:
        mf.disp = 'd3zero'
    mf.conv_tol = 1e-6
    # mf.damp = 0.2
    mf.max_cycle = 200
    mf.max_memory = 32000
    mf.run()

    force = None
    force_rms = np.nan
    if mf.converged and f:
        force = mf.nuc_grad_method().kernel() * -1. / BOHR
        force_rms = np.sqrt(np.mean(force ** 2)) * AU2EV
        print("force rms (ev/A): ", force_rms)

    hessian = None
    if hess:
        hessian = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mf.mol, hessian)
        print("freq: ", freq_info['freq_wavenumber'])

    if return_metrics:
        return mf, force, hessian, force_rms, count_negative_eig(freq_info['freq_wavenumber'])
    return mf, force, hessian


# def compute_rmsd_with_optgeom(mf, transition=False, xyzfile=None):
#     e_generated = mf.e_tot
#     mol_eq = optimize(mf, maxsteps=300, transition=transition)

#     if xyzfile is None:
#         xyzfile = ".opt.xyz" if not transition else ".ts.xyz"
#     mf.mol.tofile(".tmp.xyz", format="xyz")
#     mol_eq.tofile(xyzfile, format="xyz")

#     mf_eq, _, _ = compute_efh(xyzfile, f=False, hess=False)
#     e_eq = mf_eq.e_tot
#     e_diff = (e_eq - e_generated) * AU2KCALMOL

#     rmsd = pymatgen_rmsd(
#         mol1=".tmp.xyz",
#         mol2=xyzfile,
#         ignore_chirality=True,
#         threshold=0.5,
#     )

#     return rmsd, e_diff


# def compute_irc(mf, hessian, ts_xyz=".ts.xyz", dq=0.1):
#     ms_mol_eq = mol3D()
#     ms_mol_eq.readfromxyz(ts_xyz)
#     freq_info = thermo.harmonic_analysis(mf.mol, hessian)

#     # Left
#     new_coords = ms_mol_eq.coordsvect() - freq_info["norm_mode"][0] * dq
#     for ii, atom in enumerate(ms_mol_eq.atoms):
#         atom.setcoords(new_coords[ii])
#     ms_mol_eq.writexyz("ts-.xyz")
#     _mf, _, _ = compute_efh(
#         "ts-.xyz", hess=False, return_metrics=False)
#     _, _ = compute_rmsd_with_optgeom(_mf, transition=False, xyzfile="opt-.xyz")

#     # Right
#     new_coords = ms_mol_eq.coordsvect() + 2 * freq_info["norm_mode"][0] * dq
#     for ii, atom in enumerate(ms_mol_eq.atoms):
#         atom.setcoords(new_coords[ii])
#     ms_mol_eq.writexyz("ts+.xyz")
#     _mf, _, _ = compute_efh(
#         "ts+.xyz", hess=False, return_metrics=False)
#     _, _ = compute_rmsd_with_optgeom(_mf, transition=False, xyzfile="opt+.xyz")


def compute_barrier(opt1_xyz, ts_xyz, opt2_xyz):
    """Barrier heights in eV from each of the two optimised endpoints up to the transition state."""
    mf_1, _, _ = compute_efh(opt1_xyz, f=True, hess=False, return_metrics=False)
    mf_2, _, _ = compute_efh(opt2_xyz, f=True, hess=False, return_metrics=False)
    mf_ts, _, _ = compute_efh(ts_xyz, f=True, hess=False, return_metrics=False)
    barrier_left = (mf_ts.e_tot - mf_1.e_tot) * AU2EV
    barrier_right = (mf_ts.e_tot - mf_2.e_tot) * AU2EV
    return barrier_left, barrier_right


def calc_deltaE(xyzfile1, xyzfile2, f=False, xc="wb97x", basis="631g*", d3=False):
    """Single-point energy of the second geometry minus the first, in kcal/mol."""
    mf_1, _, _ = compute_efh_gpu(xyzfile1, f=f, hess=False, return_metrics=False, xc=xc, basis=basis, d3=d3)
    mf_2, _, _ = compute_efh_gpu(xyzfile2, f=f, hess=False, return_metrics=False, xc=xc, basis=basis, d3=d3)
    return (mf_2.e_tot - mf_1.e_tot) * AU2KCALMOL