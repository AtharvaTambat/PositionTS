#### Installation
To install, run:
```
$ git clone https://gitlab.com/matschreiner/neuralneb
$ cd neuralneb
$ pip install -e .
```


#### Training PaiNN model
To train a PaiNN model on the QM9x dataset start by downloading the qm9x.db by running
```
$ python scripts/download_qm9x.py {path}
```
qm9x.db will be downloaded to {path}/qm9x.db. By default the path is data/qm9x.db

Now that the dataset has been downloaded, a painn model can be trained by running
```
$ python scripts/train.py
```

To train on the Transition1x instead, download Transition1x and create an ase database from the .h5 file by following the instructions in the [Transition1x repository](https://gitlab.com/matschreiner/Transition1x).

The datasplits used for Transition1x in the NeuralNEB paper can be downloaded from:

```
data/t1x_splits.tar.gz
```
Trained models from paper can be found in 

```
data/models.tar.gz
```

and are organied such that the model data/models/painn{i}.pt is trained on data/splits/split{i}.

### Running NEB
To run NEB on a product/reactant pair run
```
python scripts/neb.py {product} {reactant} --model {model}
```
where product and reactant should both be .xyz files.
by default product and reactant are data/p.xyz, and data/r.xyz, respectively and is an example reaction, and the default model is data/painn.sd.


### Note
There has been a wealth of infrastructure for this project. This is a simplified version of the NEB and Training routines.
Both scripts are kept barebones such that any features can be implemented esasily by the user.
