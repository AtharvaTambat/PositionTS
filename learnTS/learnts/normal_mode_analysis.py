# This code is from https://gist.github.com/craldaz/b38e1c951d515c807c67aac303406343
import numpy as np
from copy import deepcopy
#from file_utilities import read_xyz,get_atoms,xyz_to_np,np_to_xyz,write_xyzs
#import units
#import elements
import os

def eckart_frame(                                                                                                                                                                                                                                             
    pos,
    masses,
    ):

    """ Moves the molecule to the Eckart frame
    Params:
        geom ((natoms,4) np.ndarray) - Contains atom symbol and xyz coordinates
        masses ((natoms) np.ndarray) - Atom masses
    Returns:
        COM ((3), np.ndarray) - Molecule center of mess
        L ((3), np.ndarray) - Principal moments
        O ((3,3), np.ndarray)- Principle axes of inertial tensor
        geom2 ((natoms,4 np.ndarray) - Contains new geometry (atom symbol and xyz coordinates)
    """

    # Center of mass
    COM = np.sum(pos * np.outer(masses, [1.0]*3), 0) / np.sum(masses)
    # Inertial tensor
    I = np.zeros((3,3))
    for p, mass in zip(pos, masses):
        I[0,0] += mass * (p[0] - COM[0]) * (p[0] - COM[0])
        I[0,1] += mass * (p[0] - COM[0]) * (p[1] - COM[1])
        I[0,2] += mass * (p[0] - COM[0]) * (p[2] - COM[2])
        I[1,0] += mass * (p[1] - COM[1]) * (p[0] - COM[0])
        I[1,1] += mass * (p[1] - COM[1]) * (p[1] - COM[1])
        I[1,2] += mass * (p[1] - COM[1]) * (p[2] - COM[2])
        I[2,0] += mass * (p[2] - COM[2]) * (p[0] - COM[0])
        I[2,1] += mass * (p[2] - COM[2]) * (p[1] - COM[1])
        I[2,2] += mass * (p[2] - COM[2]) * (p[2] - COM[2])
    I /= np.sum(masses)
    # Principal moments/Principle axes of inertial tensor
    L, O = np.linalg.eigh(I)

    # Eckart geometry
    pos2= np.dot((pos - np.outer(np.ones((len(masses),)), COM)), O)

    return COM, L, O, pos2


def vibrational_basis(
    pos,
    masses,
    ):  

    """ Compute the vibrational basis in mass-weighted Cartesian coordinates.
    This is the column-space of the translations and rotations in the Eckart frame.
    
    Params: 
        geom (geometry struct) - 
        masses (list of float) - masses for the geometry
    Returns:
        B ((3*natom, 3*natom-6) np.ndarray) - orthonormal basis for vibrations. 
        Mass-weighted cartesians in rows, mass-weighted vibrations in columns. 
    """
                                                                                                                                                                                                                                                              
    # Compute Eckart frame geometry
    # L,O are the Principle moments/Principle axes of the intertial tensor
    COM, L, O, pos2 = eckart_frame(pos, masses)
    G = pos2

    # Known basis functions for translations
    TR = np.zeros((3*len(pos),6))
    # Translations
    TR[0::3,0] = np.sqrt(masses) # +X
    TR[1::3,1] = np.sqrt(masses) # +Y
    TR[2::3,2] = np.sqrt(masses) # +Z

    # Rotations in the Eckart frame
    for A, mass in enumerate(masses):
        mass_12 = np.sqrt(mass)
        for j in range(3):
            TR[3*A+j,3] = + mass_12 * (G[A,1] * O[j,2] - G[A,2] * O[j,1]) # + Gy Oz - Gz Oy 
            TR[3*A+j,4] = - mass_12 * (G[A,0] * O[j,2] - G[A,2] * O[j,0]) # - Gx Oz + Gz Ox 
            TR[3*A+j,5] = + mass_12 * (G[A,0] * O[j,1] - G[A,1] * O[j,0]) # + Gx Oy - Gy Ox 

    #print(f'TR is {TR}')

    # Single Value Decomposition      
    U, s, V = np.linalg.svd(TR, full_matrices=True)

    # The null-space of TR
    B = U[:,6:]
    return B


def normal_modes(
        pos,       # Optimized geometry in au 
        hess,       # Hessian matrix in au
        masses,     # Masses in au 
        ):
    """
    Params:
        geom ((natoms,4) np.ndarray) - atoms symbols and xyz coordinates
        hess ((natoms*3,natoms*3) np.ndarray) - molecule hessian
        masses ((natoms) np.ndarray) - masses
    Returns:
        w ((natoms*3 - 6) np.ndarray)  - normal frequencies
        Q ((natoms*3, natoms*3 - 6) np.ndarray)  - normal modes
    """

    # masses repeated 3x for each atom (unravels)
    m = np.ravel(np.outer(masses,[1.0]*3))

    # mass-weight hessian
    hess2 = hess / np.sqrt(np.outer(m,m))

    # Find normal modes (project translation/rotations before)
    B = vibrational_basis(pos, masses)
    
    h, U3 = np.linalg.eigh(np.dot(B.T,np.dot(hess2,B)))
    U = np.dot(B, U3)
    # U = (3N,3N-6)

    # Normal frequencies
    v = np.sqrt(h)
    # Imaginary frequencies
    v[h < 0.0] = -np.sqrt(-h[h < 0.0])
    
    # Normal modes
    Q = U / np.outer(np.sqrt(m), np.ones((U.shape[1],)))

    return v, Q 

if __name__=='__main__':
    from ase.data import atomic_masses, atomic_numbers
    import sys
    sys.path.append('/home/shchoi/activation_energies/cgrambow-ard_gsm-a3141a8')
    from ard_gsm.qchem import QChem
    from extract_Hessian import read_log

    i=0
    directory=  'wb97xd3/'
    case = 'p'
    symbols, pos = QChem(logfile=directory+"/rxn"+str(i).zfill(6)+"/"+case+str(i).zfill(6)+".log").get_geometry()
    hess = read_log(directory+"/rxn"+str(i).zfill(6)+"/"+case+str(i).zfill(6)+".log", len(symbols))

    masses = list(map(lambda s: atomic_masses[atomic_numbers[s]], symbols))
    w, Q = normal_modes(pos, hess, masses)
    indices = np.argsort(w)
    w = w[indices]
    Q = Q[:, indices]
    print(w[0])
    print(Q[:, 0])
