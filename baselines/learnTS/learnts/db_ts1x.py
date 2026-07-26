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
sys.path.append('../ard_gsm')
symbol2number = {'H':0, 'C':1, 'N':2, 'O':3, 'F':4}

num_images = 1
#number2number = {1:0, 6:1, 7:2, 8:3, 9:4}


def get_weight(symbols):
    masses = list(map(lambda s: atomic_masses[atomic_numbers[s]], symbols))
    # print(masses)
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
    def __init__(self, directory): #, db_normalizer, use_fit ):
        from functools import reduce
        import re
        from ase.data import atomic_numbers as get_atomic_number
        #import json
        from multiprocessing import Pool
        import functools
        from ard_gsm.qchem import QChem
        from functools import partial
        import time
        import pickle
        atom_map = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
#        assert len(db_normalizer)==3
#        self.db_normalizer = db_normalizer


        #st = time.time()
        #et = time.time()
        #print("qchem: ",et-st)
        #st = time.time()
        with open(directory, "rb") as f:
            data = pickle.load(f)

        self.list_positions_r = []
        self.list_positions_p = []
        self.list_positions_ts= []
        self.list_symbols_r = []
        self.list_symbols_p = []
        self.list_symbols_ts= []
        self.list_e_r = []
        self.list_e_p = []
        self.list_e_ts= []
        for i in range(len(data)):
            #coordinates
            self.list_positions_r.append( np.array(data[i]['pos_0']) )
            self.list_positions_p.append( np.array(data[i]['pos_2']) )
            self.list_positions_ts.append( np.array(data[i]['pos_1']) )   

            #symbols
            self.list_symbols_r.append([atom_map[x.item()] for x in data[i]['charge_0']])
            self.list_symbols_p.append([atom_map[x.item()] for x in data[i]['charge_2']])
            self.list_symbols_ts.append([atom_map[x.item()] for x in data[i]['charge_1']])
            #energies
            self.list_e_r.append(np.array(data[i]['energy_0']))
            self.list_e_p.append(np.array(data[i]['energy_2']))
            self.list_e_ts.append(np.array(data[i]['energy_1']))

        #weights
        self.weight = list(map(get_weight, self.list_symbols_r))
        #filling the rest with zeros same length as data
        self.list_s_r = [np.zeros(2, dtype=np.float32) for _ in range(len(data))]
        self.list_s_p = [np.zeros(2, dtype=np.float32) for _ in range(len(data))]
        self.list_s_ts = [np.zeros(2, dtype=np.float32) for _ in range(len(data))]
        # self.hess_r = np.zeros((len(self.weight), 40, 40))
        # self.hess_p = np.zeros((len(self.weight), 40, 40))
        # self.hess_ts = np.zeros((len(self.weight), 40, 40))
        # self.B_r = np.zeros((len(self.weight), 40, 40))
        # self.B_p = np.zeros((len(self.weight), 40, 40))
        # self.B_ts = np.zeros((len(self.weight), 40, 40))





        self.list_images = list(map(calculate_images, zip(self.list_positions_r, self.list_positions_p, self.list_positions_ts) ))
        self.list_linear_images = list(map(calculate_images, zip(self.list_positions_r, self.list_positions_p) ))
        self.list_edm =  list(map( calculate_edm , self.list_images ))
        self.list_linear_edm =  list(map( calculate_edm , self.list_linear_images ))

        self.list_symbols = self.list_symbols_r
        self.length = len(self.list_symbols)
        self.num_images = num_images
    def __len__(self,):
        return self.length

    def __getitem__(self, idx):
        num_atoms = len(self.list_symbols[idx])

        B_zeros = np.zeros((3, 3*num_atoms, 3*num_atoms-6), dtype=np.float32)
        hess_zeros = np.zeros((3, 3*num_atoms-6, 3*num_atoms-6), dtype=np.float32)

        return np.fromiter((symbol2number[x] for x in self.list_symbols[idx]), dtype=np.int64, count=num_atoms), \
            self.list_edm[idx].astype(np.float32), \
            self.list_linear_edm[idx].astype(np.float32), \
            np.float32(self.list_e_ts[idx] - self.list_e_r[idx]), \
            np.float32(self.list_e_ts[idx] - self.list_e_p[idx]), \
            (self.list_s_ts[idx] - self.list_s_r[idx]).astype(np.float32), \
            (self.list_s_ts[idx] - self.list_s_p[idx]).astype(np.float32), \
            B_zeros, \
            hess_zeros, \
            self.weight[idx].astype(np.float32)

                #true path edm
                # linear path edm
                #forward barrier
                #reverse barrier
                #forward entropy diff
                #reverse entropy diff
                # B matrices for r, ts, p
                # hessian for r, ts, p
                #mass-weight matrix

#               self.db_normalizer[0].transform(self.list_edm_r[idx].reshape(-1,1) ).reshape(self.list_edm_r[idx].shape),  \
#               self.db_normalizer[1].transform(self.list_edm_p[idx].reshape(-1,1) ).reshape(self.list_edm_p[idx].shape),  \
#               self.db_normalizer[2].transform(self.list_edm_ts[idx].reshape(-1,1)).reshape(self.list_edm_ts[idx].shape)

def my_collate_fn(batch, num_images, grid, alpha, duplicate=False):


    factor = 2 if (duplicate) else 1
    batch_size = len(batch)
    max_atoms  = max( [ len(item[0]) for item in batch ] )

    atomic_numbers = np.zeros((factor*batch_size, max_atoms ), dtype=int)
    true_edm  = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms ), dtype=np.float32)
    linear_edm  = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms ), dtype=np.float32)
    energy_r = np.zeros((factor*batch_size ) )
    energy_p = np.zeros((factor*batch_size ) )

    entropy_r = np.zeros((factor*batch_size,2 ) )
    entropy_p = np.zeros((factor*batch_size,2 ) )

    B_matrices = np.zeros((factor*batch_size, 3, 3*max_atoms, 3*max_atoms-6), dtype=np.float32)
    hess = np.zeros((factor*batch_size, 3, 3*max_atoms-6, 3*max_atoms-6), dtype=np.float32)
    inv_weights  = np.zeros((factor*batch_size, 3*max_atoms, 3*max_atoms ), dtype=np.float32)

    edm_mask = np.zeros((factor*batch_size, 2*num_images+1, max_atoms, max_atoms), dtype=bool)
    edm_diag_mask = np.ones((factor*batch_size, 2*num_images+1, max_atoms, max_atoms), dtype=bool)
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
            #entropy not reversed
            entropy_r[batch_size+i] = entropy_p_
            entropy_p[batch_size+i] = entropy_r_
            # entropy_r[batch_size+i] = entropy_r_
            # entropy_p[batch_size+i] = entropy_p_

            B_matrices[batch_size+i,:,:3*num_atoms, :3*num_atoms-6] = B_
            hess[batch_size+i, :, :3*num_atoms-6,:3*num_atoms-6] = hess_
            # inv_weights[i,:3*num_atoms, :3*num_atoms] = weights_
            inv_weights[batch_size+i,:3*num_atoms, :3*num_atoms] = weights_

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

    #not multiplying by 627.5
    return torch.from_numpy(atomic_numbers).type(torch.long),\
           torch.from_numpy(true_edm).type(torch.float32),\
           torch.from_numpy(linear_edm).type(torch.float32),\
           torch.from_numpy(edge_f).type(torch.float32),\
           torch.from_numpy(energy_r).type(torch.float32),\
           torch.from_numpy(energy_p).type(torch.float32),\
           torch.from_numpy(entropy_r).type(torch.float32),\
           torch.from_numpy(entropy_p).type(torch.float32),\
           torch.from_numpy(B_matrices).type(torch.float32),\
           torch.from_numpy(hess).type(torch.float32),\
           torch.from_numpy(inv_weights).type(torch.float32),\
           torch.from_numpy(edm_mask).type(torch.bool),\
           torch.from_numpy(edm_diag_mask).type(torch.bool)
    # return torch.from_numpy(atomic_numbers).type(torch.long),\
    #        torch.from_numpy(true_edm).type(torch.float32),\
    #        torch.from_numpy(linear_edm).type(torch.float32),\
    #        torch.from_numpy(edge_f).type(torch.float32),\
    #        torch.from_numpy(energy_r).type(torch.float32)*627.5096080305927,\
    #        torch.from_numpy(energy_p).type(torch.float32)*627.5096080305927,\
    #        torch.from_numpy(entropy_r).type(torch.float32),\
    #        torch.from_numpy(entropy_p).type(torch.float32),\
    #        torch.from_numpy(B_matrices).type(torch.float32),\
    #        torch.from_numpy(hess).type(torch.float32),\
    #        torch.from_numpy(inv_weights).type(torch.float32),\
    #        torch.from_numpy(edm_mask).type(torch.bool),\
    #        torch.from_numpy(edm_diag_mask).type(torch.bool)

if __name__=="__main__":
    import pickle
    import argparse
    parser = argparse.ArgumentParser()
    # split directory, e.g. "../../data/rgdy_subset_10_splits/"
    parser.add_argument("--directory", type=str, default="")

    args = parser.parse_args()
    
    directory = args.directory
    train_dir = directory + "train_split.pkl"
    valid_dir = directory + "validation_split.pkl"
    test_dir = directory + "test_split.pkl"


    print("Creating valid-db-distance_rgdy_10.pkl")
    with open("valid-db-distance_rgdy_10.pkl","wb") as f:
        db= DB(valid_dir)
        pickle.dump( db, f)

    print("Creating test-db-distance_rgdy_10.pkl")
    with open("test-db-distance_rgdy_10.pkl","wb") as f:
        db=DB(test_dir)
        pickle.dump( db, f)

    print("Creating train-db-distance_rgdy_10.pkl")
    with open("train-db-distance_rgdy_10.pkl","wb") as f:
        db=DB(train_dir)
        pickle.dump( db, f)

    e_p_all = np.array(db.list_e_ts) - np.array(db.list_e_p )
    e_r_all = np.array(db.list_e_ts) - np.array(db.list_e_r )

    e_all = np.concatenate([e_p_all, e_r_all]) #*627.5096080305927 no need to multiply
    print(e_all.mean(), e_all.std())

    s_p_all = np.array(db.list_s_ts) - np.array(db.list_s_p )
    s_r_all = np.array(db.list_s_ts) - np.array(db.list_s_r )

    s_all = np.concatenate([s_p_all, s_r_all])
    print(np.mean( s_all, axis=0 ), np.std(s_all, axis=0) )
