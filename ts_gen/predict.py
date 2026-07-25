# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
from rdkit import Chem

from model.G2C import G2C


# -----------------------------
# Load SDF
# -----------------------------
def load_sdf(path):
    from rdkit import Chem

    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)

    mols = []
    for i, mol in enumerate(supplier):
        if mol is None:
            print("Failed to read molecule {}".format(i))
            continue

        # Same logic as notebook — just properly structured
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol, addCoords=True)

        mols.append(mol)

    return mols


# -----------------------------
# Prepare features (UNCHANGED)
# -----------------------------
def prepare_batch(reactants, products, MAX_SIZE, elements):
    num_elements = len(elements)
    size = len(reactants)

    V = np.zeros((size, MAX_SIZE, num_elements+1), dtype=np.float32)
    E = np.zeros((size, MAX_SIZE, MAX_SIZE, 3), dtype=np.float32)
    sizes = np.zeros(size, dtype=np.int32)
    coordinates = np.zeros((size, MAX_SIZE, 3), dtype=np.float32)

    for bx in range(size):
        r = reactants[bx]
        p = products[bx]

        N = r.GetNumAtoms()
        sizes[bx] = N

        print("Reactant atoms list:", [a.GetSymbol() for a in r.GetAtoms()])
        print("Product atoms list :", [a.GetSymbol() for a in p.GetAtoms()])
        
        D = (Chem.GetDistanceMatrix(r) + Chem.GetDistanceMatrix(p)) / 2
        D[D > 10.] = 10.

        D3 = np.exp(-(Chem.Get3DDistanceMatrix(r) +
                      Chem.Get3DDistanceMatrix(p)) / 2)

        for i in range(N):
            for j in range(N):
                E[bx, i, j, 2] = D3[i][j]
                if D[i][j] == 1.:
                    E[bx, i, j, 1] = 1.

            pos = r.GetConformer().GetAtomPosition(i)
            coordinates[bx, i] = [pos.x, pos.y, pos.z]

            atom = r.GetAtomWithIdx(i)
            if atom.GetSymbol() in elements:
                idx = elements.index(atom.GetSymbol())
                V[bx, i, idx] = 1.
            V[bx, i, num_elements] = atom.GetAtomicNum() / 10.

    return {
        "nodes": V,
        "edges": E,
        "sizes": sizes,
        "coordinates": coordinates
    }


# -----------------------------
# Write XYZ
# -----------------------------
def write_xyz(path, mol, coords, idx):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]

    with open(path, "w") as f:
        f.write("{}\nTS index {}\n".format(len(atoms), idx))
        for a, c in zip(atoms, coords):
            f.write("{} {} {} {}\n".format(a, c[0], c[1], c[2]))


# -----------------------------
# Predict
# -----------------------------
def predict(react_sdf, prod_sdf, ckpt, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reactants = load_sdf(react_sdf)
    products  = load_sdf(prod_sdf)

    assert len(reactants) == len(products), "Mismatch in dataset size"

    print("[INFO] Loaded {} reactions".format(len(reactants)))

    MAX_SIZE = max([m.GetNumAtoms() for m in reactants])
    elements = "HCNO"

    model = G2C(
        max_size=MAX_SIZE,
        node_features=len(elements)+1,
        edge_features=3,
        layers=2,
        hidden_size=128,
        iterations=3
    )

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt)

        print("[INFO] Model restored from {}".format(ckpt))

        for i in range(len(reactants)):

            batch = prepare_batch(
                [reactants[i]],
                [products[i]],
                MAX_SIZE,
                elements
            )

            feed_dict = {
                model.placeholders[k]: batch[k]
                for k in batch
            }

            X = sess.run(model.tensors["X"], feed_dict=feed_dict)[0]

            coords = X[:reactants[i].GetNumAtoms()]

            out_file = os.path.join(output_dir, "tsgen_idx_{:06d}.xyz".format(i))
            write_xyz(out_file, reactants[i], coords, i)

            # logging
            if (i + 1) % 10 == 0 or i == len(reactants) - 1:
                print("[PROGRESS] {}/{} TS generated".format(i+1, len(reactants)))


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    react_sdf = sys.argv[1]
    prod_sdf  = sys.argv[2]
    ckpt      = sys.argv[3]
    out_dir   = sys.argv[4]

    predict(react_sdf, prod_sdf, ckpt, out_dir)