# -*- coding: utf-8 -*-
"""Predict transition-state geometries from a trained ts_gen G2C checkpoint.

Current entry point (--reactants/--products/--checkpoint/--output_dir); predict.py is
the older superseded version.
"""

from __future__ import print_function

import os
import argparse
import numpy as np
import tensorflow as tf
from rdkit import Chem, Geometry

from model.G2C import G2C


# ---------- LOAD DATA ----------
def load_data(reactant_file, product_file):
    """Pair the two SDFs into (reactant, product) tuples, dropping any pair either supplier fails to read."""
    r_suppl = Chem.ForwardSDMolSupplier(reactant_file, removeHs=False, sanitize=False)
    p_suppl = Chem.ForwardSDMolSupplier(product_file, removeHs=False, sanitize=False)

    data = []
    for r, p in zip(r_suppl, p_suppl):
        if r is not None and p is not None:
            data.append((r, p))
    return data


# ---------- PREPARE BATCH (FROM NOTEBOOK) ----------
def prepare_batch(batch_mols, MAX_SIZE, elements):
    """Build the padded node, edge, size and coordinate arrays G2C is fed, keyed "V"/"E"/"sizes"/"coordinates".

    Edge channel 0 is the mean topological distance capped at 10, channel 1 exp(-mean 3D
    distance), channel 2 unused. Coordinates come from the reactant conformer.
    """

    num_elements = len(elements)

    size = len(batch_mols)
    V = np.zeros((size, MAX_SIZE, num_elements+1), dtype=np.float32)
    E = np.zeros((size, MAX_SIZE, MAX_SIZE, 3), dtype=np.float32)
    sizes = np.zeros(size, dtype=np.int32)
    coordinates = np.zeros((size, MAX_SIZE, 3), dtype=np.float32)

    for bx in range(size):
        reactant, product = batch_mols[bx]
        N_atoms = reactant.GetNumAtoms()
        sizes[bx] = int(N_atoms)

        MAX_D = 10.
        D = (Chem.GetDistanceMatrix(reactant) + Chem.GetDistanceMatrix(product)) / 2
        D[D > MAX_D] = 10.

        D_3D_rbf = np.exp(-((Chem.Get3DDistanceMatrix(reactant) + Chem.Get3DDistanceMatrix(product)) / 2))

        for i in range(N_atoms):

            atom = reactant.GetAtomWithIdx(i)
            symbol = atom.GetSymbol()

            if symbol in elements:
                V[bx, i, elements.index(symbol)] = 1
            else:
                V[bx, i, -1] = 1

            for j in range(N_atoms):
                E[bx, i, j, 0] = D[i, j]
                E[bx, i, j, 1] = D_3D_rbf[i, j]

        conf = reactant.GetConformer()
        for i in range(N_atoms):
            pos = conf.GetAtomPosition(i)
            coordinates[bx, i, :] = [pos.x, pos.y, pos.z]

    return {
        "V": V,
        "E": E,
        "sizes": sizes,
        "coordinates": coordinates
    }


# ---------- BATCH GENERATOR ----------
def sample_batch(data, BATCH_SIZE, MAX_SIZE, elements):
    """Yield (feature dict, molecule pairs) for each consecutive slice of BATCH_SIZE reactions."""
    total = len(data)
    for i in range(0, total, BATCH_SIZE):
        batch_mols = data[i:i+BATCH_SIZE]
        batch_dict = prepare_batch(batch_mols, MAX_SIZE, elements)
        yield batch_dict, batch_mols


# ---------- SAVE XYZ ----------
def save_xyz(coords, mol, filepath):
    f = open(filepath, "w")

    N = mol.GetNumAtoms()
    f.write(str(N) + "\n\n")

    for i in range(N):
        atom = mol.GetAtomWithIdx(i)
        symbol = atom.GetSymbol()

        x = coords[i][0]
        y = coords[i][1]
        z = coords[i][2]

        f.write("{} {:.6f} {:.6f} {:.6f}\n".format(symbol, x, y, z))

    f.close()


# ---------- MAIN ----------
def main(args):

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    print("Loading data...")
    test_data = load_data(args.reactants, args.products)

    MAX_SIZE = max([x.GetNumAtoms() for x, y in test_data])
    elements = "HCNO"
    num_elements = len(elements)

    print("Building model...")
    model = G2C(
        max_size=MAX_SIZE,
        node_features=num_elements+1,
        edge_features=3,
        layers=2,
        hidden_size=128,
        iterations=3
    )

    config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Session(config=config) as sess:

        print("Restoring model...")
        saver = tf.train.Saver()
        saver.restore(sess, args.checkpoint)

        generator = sample_batch(test_data, 1, MAX_SIZE, elements)

        print("Running inference...")

        idx = 0
        for step, data in enumerate(generator):

            batch_dict, batch_mols = data

            mapped_batch = {
                "nodes": batch_dict["V"],
                "edges": batch_dict["E"],
                "coordinates": batch_dict["coordinates"],
                "sizes": batch_dict["sizes"]
            }

            feed_dict = {}
            for key in model.placeholders:
                feed_dict[model.placeholders[key]] = mapped_batch[key]

            X = sess.run(model.tensors["X"], feed_dict=feed_dict)

            # Save each molecule
            for b in range(len(batch_mols)):
                mol = batch_mols[b][0]  # reactant template

                coords = X[b]

                out_file = os.path.join(args.output_dir, "tsgen_idx_{}.xyz".format(idx))
                save_xyz(coords, mol, out_file)

                idx += 1

            if step % 100 == 0:
                print("Processed batch", step)


# ---------- CLI ----------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--reactants", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="ts_xyz")

    args = parser.parse_args()

    main(args)