import os
import re
import torch
from torch.utils.data import Dataset
from ase import Atoms, Atom
from ase.data import atomic_masses, atomic_numbers
import numpy as np
from functools import lru_cache
from tqdm import tqdm
import sys
sys.path.append('/data/shchoi/activation_learning/trajectory/ard_gsm/')
sys.path.append('/home/shchoi/activation_energies/cgrambow-ard_gsm-a3141a8/')
symbol2number = {'H':0, 'C':1, 'N':2, 'O':3, 'F':4}
num_images = 1
#number2number = {1:0, 6:1, 7:2, 8:3, 9:4}


def create_qchem(i, directory, case):
    from ard_gsm.qchem import QChem
    return QChem(logfile=directory+"/rxn"+str(i).zfill(6)+"/"+case+str(i).zfill(6)+".log")
def get_geom(qchem):
    return qchem.get_geometry()[1]
def get_symbol(qchem):
    return qchem.get_geometry()[0]
def get_e(qchem):
    return qchem.get_energy()

def get_s(qchem):
    from extract_Hessian import read_entropy
    return read_entropy( qchem.logfile )

def get_B(qchem):
    from normal_mode_analysis import vibrational_basis
    symbols, pos = qchem.get_geometry()
    masses = list(map(lambda s: atomic_masses[atomic_numbers[s]], symbols))
    return vibrational_basis(pos, masses)
def get_hess(qchem):
    from extract_Hessian import read_hess
    from normal_mode_analysis import vibrational_basis
    symbols, pos = qchem.get_geometry()
    hess = read_hess(qchem.logfile, len(symbols))
    masses = list(map(lambda s: atomic_masses[atomic_numbers[s]], symbols))
    B= vibrational_basis(pos, masses)
    m = np.ravel(np.outer(masses,[1.0]*3))
    hess2 = hess / np.sqrt(np.outer(m,m))
    return np.dot(B.T,np.dot(hess2,B))

def get_weight(qchem):
    symbols, pos = qchem.get_geometry()
    masses = list(map(lambda s: atomic_masses[atomic_numbers[s]], symbols))
    m = np.ravel(np.outer(masses,[1.0]*3))
    return np.sqrt(np.outer(m,m))

def calculate_images(inp):
    out = []
    if(len(inp)==3):
        positions_r, positions_p, positions_ts = inp
        for i in range(num_images):
            factor = i / num_images
            out.append( positions_r * (1-factor) + positions_ts *factor )
        for i in range(num_images):
            factor = i / num_images
            out.append( positions_ts * (1-factor) + positions_p *factor )
    elif(len(inp)==2):
        positions_r, positions_p = inp
        for i in range(2*num_images):
            factor = i / (2*num_images)
            out.append( positions_r * (1-factor) + positions_p *factor )
    else:
        raise RuntimeError

    out.append(positions_p)
    return out

def calculate_edm(positions):
    out = []
    for position in positions:
        N = len(position)
        X = np.transpose(position)
        one=np.ones((N,1))
        G = np.transpose(X)@X
        diag_G = np.diag(G).reshape((N,1))
        out.append( one@diag_G.T - 2* G  + diag_G@one.T )
    return np.array(out)

class DB(Dataset):
    def __init__(self, directory, indices): #, db_normalizer, use_fit ):
        from functools import reduce
        import re
        from ase.data import atomic_numbers as get_atomic_number
        #import json
        from multiprocessing import Pool
        import functools
        from ard_gsm.qchem import QChem
        from functools import partial
        import time
#        assert len(db_normalizer)==3
#        self.db_normalizer = db_normalizer


        with Pool() as p:
            #st = time.time()
            list_qchem_r = p.map( partial(create_qchem,directory=directory, case="r"), indices)
            list_qchem_p = p.map( partial(create_qchem,directory=directory, case="p"), indices)
            list_qchem_ts= p.map( partial(create_qchem,directory=directory, case="ts"),indices)
            #et = time.time()
            #print("qchem: ",et-st)
            #st = time.time()

            self.list_positions_r = p.map( get_geom, list_qchem_r)
            self.list_positions_p = p.map( get_geom, list_qchem_p)
            self.list_positions_ts= p.map( get_geom, list_qchem_ts)

            self.list_s_r = np.array( p.map(get_s, list_qchem_r) )
            self.list_s_p = np.array( p.map(get_s, list_qchem_p) )
            self.list_s_ts= np.array( p.map(get_s, list_qchem_ts))

            self.weight = p.map(get_weight, list_qchem_r)

            self.hess_r = p.map(get_hess, list_qchem_r)
            self.hess_p = p.map(get_hess, list_qchem_p)
            self.hess_ts = p.map(get_hess, list_qchem_ts)

            self.B_r = p.map(get_B, list_qchem_r)
            self.B_p = p.map(get_B, list_qchem_p)
            self.B_ts = p.map(get_B, list_qchem_ts)

            self.list_e_r= p.map( get_e, list_qchem_r)
            self.list_e_p= p.map( get_e, list_qchem_p)
            self.list_e_ts=p.map( get_e, list_qchem_ts)
            #et = time.time()
            #print("positions: ", et-st)

            self.list_images = p.map(calculate_images, zip(self.list_positions_r, self.list_positions_p, self.list_positions_ts) )
            self.list_linear_images = p.map(calculate_images, zip(self.list_positions_r, self.list_positions_p) )
            self.list_edm =  p.map( calculate_edm , self.list_images )
            self.list_linear_edm =  p.map( calculate_edm , self.list_linear_images )
            self.list_symbols = p.map( get_symbol, list_qchem_r)

        self.length = len(self.list_symbols)
        self.num_images = num_images
    def __len__(self,):
        return self.length

    def __getitem__(self,idx):
        size = len(self.list_symbols[idx])
#        indices = torch.randperm(size)
#        return np.fromiter(map(lambda x:symbol2number[x] , self.list_symbols[idx]), np.int)[indices],  \
#               self.list_edm_r[idx][indices][:,indices],  \
#               self.list_edm_p[idx][indices][:,indices],  \
#               self.list_edm_ts[idx][indices][:,indices]
        return np.fromiter(map(lambda x:symbol2number[x] , self.list_symbols[idx]), np.int),  \
               self.list_edm[idx],\
               self.list_linear_edm[idx],\
               self.list_e_ts[idx]-self.list_e_r[idx],\
               self.list_e_ts[idx]-self.list_e_p[idx],\
               self.list_s_ts[idx]-self.list_s_r[idx],\
               self.list_s_ts[idx]-self.list_s_p[idx],\
               np.stack([self.B_r[idx], self.B_ts[idx], self.B_p[idx]]),\
               np.stack([self.hess_r[idx], self.hess_ts[idx], self.hess_p[idx]]),\
               self.weight[idx]


#               self.db_normalizer[0].transform(self.list_edm_r[idx].reshape(-1,1) ).reshape(self.list_edm_r[idx].shape),  \
#               self.db_normalizer[1].transform(self.list_edm_p[idx].reshape(-1,1) ).reshape(self.list_edm_p[idx].shape),  \
#               self.db_normalizer[2].transform(self.list_edm_ts[idx].reshape(-1,1)).reshape(self.list_edm_ts[idx].shape)

def my_collate_fn(batch, num_images, grid, alpha, duplicate=False):


    factor = 2 if (duplicate) else 1
    batch_size = len(batch)
    max_atoms  = max( [ len(item[0]) for item in batch ] )

    atomic_numbers = np.zeros((factor*batch_size, max_atoms ), dtype=np.int)
    true_edm  = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms ), dtype=np.float32)
    linear_edm  = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms ), dtype=np.float32)
    energy_r = np.zeros((factor*batch_size ) )
    energy_p = np.zeros((factor*batch_size ) )

    entropy_r = np.zeros((factor*batch_size,2 ) )
    entropy_p = np.zeros((factor*batch_size,2 ) )

    B_matrices = np.zeros((factor*batch_size, 3, 3*max_atoms, 3*max_atoms-6), dtype=np.float32)
    hess = np.zeros((factor*batch_size, 3, 3*max_atoms-6, 3*max_atoms-6), dtype=np.float32)
    inv_weights  = np.zeros((factor*batch_size, 3*max_atoms, 3*max_atoms ), dtype=np.float32)

    edm_mask = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms), dtype=np.bool)
    edm_diag_mask = np.ones((factor*batch_size, 2*num_images+1, max_atoms, max_atoms), dtype=np.bool)
    edm_diag_mask[:,:, np.arange(max_atoms), np.arange(max_atoms)] =False

#    atomic_numbers = np.zeros((2*batch_size, max_atoms ), dtype=np.int)
#    edm_r  = np.zeros((2*batch_size, max_atoms, max_atoms ), dtype=np.float32)
#    edm_p  = np.zeros((2*batch_size, max_atoms, max_atoms ), dtype=np.float32)
#    edm_ts = np.zeros((2*batch_size, max_atoms, max_atoms ), dtype=np.float32)
#    mask = np.zeros((2*batch_size, max_atoms), dtype=np.bool)
#
#    target = np.zeros((2*batch_size, max_atoms, max_atoms))

    for i, (atomic_numbers_, edm_, linear_edm_, e_r_, e_p_, entropy_r_, entropy_p_, B_, hess_, weights_) in enumerate(batch):
        num_atoms = len(atomic_numbers_)
        atomic_numbers [i,:num_atoms] = atomic_numbers_
        true_edm [i,:, :num_atoms, :num_atoms] = edm_
        linear_edm [i,:, :num_atoms, :num_atoms] = linear_edm_
        energy_r[i] = e_r_
        energy_p[i] = e_p_
        entropy_r[i] = entropy_r_
        entropy_p[i] = entropy_p_

        B_matrices[i,:,:3*num_atoms, :3*num_atoms-6] = B_
        hess[i, :, :3*num_atoms-6,:3*num_atoms-6] = hess_
        inv_weights[i,:3*num_atoms, :3*num_atoms] = weights_

        edm_mask[i,:, :num_atoms, :num_atoms] = True
    if (duplicate):
        for i, (atomic_numbers_, edm_, linear_edm_, e_r_, e_p_, entropy_r_, entropy_p_, B_, hess_, weights_) in enumerate(batch):
            num_atoms = len(atomic_numbers_)
            atomic_numbers [batch_size+i,:num_atoms] = atomic_numbers_
            true_edm [batch_size+i,:, :num_atoms, :num_atoms] = edm_[::-1]
            linear_edm [batch_size+i,:, :num_atoms, :num_atoms] = linear_edm_[::-1]
            energy_r[batch_size+i] = e_p_
            energy_p[batch_size+i] = e_r_
            entropy_r[batch_size+i] = entropy_r_
            entropy_p[batch_size+i] = entropy_p_

            B_matrices[batch_size+i,:,:3*num_atoms, :3*num_atoms-6] = B_
            hess[batch_size+i, :, :3*num_atoms-6,:3*num_atoms-6] = hess_
            inv_weights[i,:3*num_atoms, :3*num_atoms] = weights_

            edm_mask[batch_size+i,:, :num_atoms, :num_atoms] = True

    grid_ = np.reshape(grid, (1,1,1,-1) )
    triu_indices = np.triu_indices(max_atoms )

    true_edm        = np.sqrt(true_edm[:,:,triu_indices[0], triu_indices[1] ] )
    linear_edm = np.sqrt(linear_edm[:,:,triu_indices[0], triu_indices[1] ] )

    edm_mask   = edm_mask[:,:,triu_indices[0], triu_indices[1]]
    edm_diag_mask   = edm_diag_mask[:,:,triu_indices[0], triu_indices[1]]


    # hess
    triu_indices = np.triu_indices(3*max_atoms-6)
    hess = hess[:,:, triu_indices[0], triu_indices[1]]

    edge_f = np.exp( -alpha* (grid_-np.expand_dims(linear_edm,-1))**2 )
    edge_f[edm_mask==False] = 0.0

    return torch.from_numpy(atomic_numbers).type(torch.long),\
           torch.from_numpy(true_edm).type(torch.float32),\
           torch.from_numpy(linear_edm).type(torch.float32),\
           torch.from_numpy(edge_f).type(torch.float32),\
           torch.from_numpy(energy_r).type(torch.float32)*627.5096080305927,\
           torch.from_numpy(energy_p).type(torch.float32)*627.5096080305927,\
           torch.from_numpy(entropy_r).type(torch.float32),\
           torch.from_numpy(entropy_p).type(torch.float32),\
           torch.from_numpy(B_matrices).type(torch.float32),\
           torch.from_numpy(hess).type(torch.float32),\
           torch.from_numpy(inv_weights).type(torch.float32),\
           torch.from_numpy(edm_mask).type(torch.bool),\
           torch.from_numpy(edm_diag_mask).type(torch.bool)

if __name__=="__main__":
    import pickle
    directory = '/home/shchoi/activation_energies/wb97xd3/'

    test_ratio = 0.1
    valid_ratio = 0.1

    indices =list( range(0, 11961) )
    length = len(indices)
    test_size = int(length*test_ratio)
    valid_size = int(length*valid_ratio)
    train_size = length - test_size - valid_size

    print(length, test_size, valid_size, train_size)

    np.random.shuffle(indices)

    np.savetxt("db_indices.txt", indices)

    print("valid-db-distance.pkl")
    with open("valid-db-distance.pkl","wb") as f:
        db= DB(directory, indices[train_size:train_size+valid_size])
        pickle.dump( db, f)

    print("test-db-distance.pkl")
    with open("test-db-distance.pkl","wb") as f:
        db=DB(directory, indices[train_size+valid_size:])
        pickle.dump( db, f)

    print("train-db-distance.pkl")
    with open("train-db-distance.pkl","wb") as f:
        db=DB(directory, indices[:train_size])
        pickle.dump( db, f)

    e_p_all = np.array(db.list_e_ts) - np.array(db.list_e_p )
    e_r_all = np.array(db.list_e_ts) - np.array(db.list_e_r )

    e_all = np.concatenate([e_p_all, e_r_all])*627.5096080305927
    print(e_all.mean(), e_all.std())

    s_p_all = np.array(db.list_s_ts) - np.array(db.list_s_p )
    s_r_all = np.array(db.list_s_ts) - np.array(db.list_s_r )

    s_all = np.concatenate([s_p_all, s_r_all])
    print(np.mean( s_all, axis=0 ), np.std(s_all, axis=0) )
