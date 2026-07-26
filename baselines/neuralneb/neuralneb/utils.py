import numpy as np
import scipy
import torch

import ase.io
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class MLCalculator(Calculator):
    def __init__(
        self,
        model,
        implemented_properties=None,
        device=None,
        **kwargs,
    ):
        if not implemented_properties:
            implemented_properties = ["energy", "energy_var", "forces", "forces_var"]
        self.implemented_properties = implemented_properties
        pin_memory = (device == 'cuda')

        self.batch_handler = BatchHandler(pin_memory=pin_memory)

        super().__init__(**kwargs)

        # self.atoms_converter = atoms_converter
        self.model = model
        self.device = device
        if device:
            model.to(device)

    def calculate(
        self, atoms=None, properties=None, system_changes=None
    ):  # pylint:disable=unused-argument
        if isinstance(atoms, Atoms):
            atoms = [atoms]

        if not system_changes:
            system_changes = all_changes

        if not properties:
            properties = ["energy", "forces"]

        if self.calculation_required(atoms, properties):
            super().calculate(atoms)
            batch = self.batch_handler.get_batch(atoms)

            results = self.model(batch)
            energies = np.array(results["energy"].cpu().detach().numpy().squeeze(1), dtype='float64')
            forces = np.array(results["forces"].cpu().detach().numpy(), dtype='float64')

            for force, energy, atom in zip(forces, energies, atoms):
                atom.calc.results = {
                    "energy": energy.squeeze(),
                    "forces": force,
                }  # pylint:disable=attribute-defined-outside-init

                if "energy_var" in results:
                    atoms.calc.results["energy_var"] = results["energy_var"].item()
                if "forces_var" in results:
                    atoms.calc.results["forces_var"] = np.array(
                        results["forces_var"].cpu().squeeze().detach().numpy()
                    )

            for atom in atoms:
                atom.calc.atoms = atom.copy()


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def get_dataset(db, energy_key="energy", forces_key="forces"):
    dataset = AseDbData(
        db,
        TransformRowToGraphXyz(
            cutoff=5.0,
            energy_property=energy_key,
            forces_property=forces_key,
        ),
    )
    return dataset


class DummyRow():
    def __init__(self, atoms):
        self.atoms = atoms

    def toatoms(self):
        return self.atoms


class BatchHandler:
    def __init__(self, pin_memory):
        self.transform = TransformRowToGraphXyz()
        self.collate_atomsdata = CollateAtoms(pin_memory)

    def get_batch(self, atoms):
        dummyrows = [DummyRow(atom) for atom in atoms]
        graphdata = [self.transform(row) for row in dummyrows]
        return self.collate_atomsdata(graphdata)


class CollateAtoms:
    def __init__(self, pin_memory):
        self.pin_memory = pin_memory

    def __call__(self, graphs):
        dict_of_lists = {k: [dic[k] for dic in graphs] for k in graphs[0]}
        if self.pin_memory:
            def pin(x):
                if hasattr(x, "pin_memory"):
                    return x.pin_memory()
                return x
        else:
            pin = lambda x: x

        collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
        return collated


def pad_and_stack(tensors):
    """Pad list of tensors if tensors are arrays and stack if they are scalars"""
    if tensors[0].shape:
        return torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True, padding_value=0
        )
    return torch.stack(tensors)


class AseDbData(torch.utils.data.Dataset):
    def __init__(self, asedb_path, transformer, **kwargs):
        super().__init__(**kwargs)

        self.asedb_path = asedb_path
        self.asedb_connection = ase.db.connect(asedb_path)
        self.transformer = transformer

    def __len__(self):
        return len(self.asedb_connection)

    def __getitem__(self, key):
        # Note that ASE databases are 1-indexed
        try:
            return self.transformer(self.asedb_connection[key + 1])
        except KeyError:
            raise IndexError("index out of range") # pylint: disable=raise-missing-from


class TransformRowToGraphXyz:
    """
    Transform ASE DB row to graph while keeping the xyz positions of the vertices

    """

    def __init__(
        self,
        cutoff=5.0,
        energy_property="energy",
        forces_property="forces",
        energy_reference_property=None,
    ):
        self.cutoff = cutoff
        self.energy_property = energy_property
        self.forces_property = forces_property
        self.energy_reference_property = energy_reference_property

    def __call__(self, row):
        atoms = row.toatoms()

        edges, edges_displacement = self.get_edges(atoms)

        # Extract energy and forces if they exists
        try:
            energy = np.copy([np.squeeze(row.data[self.energy_property])])
        except (KeyError, AttributeError):
            energy = np.zeros(len(atoms))
        try:
            forces = np.copy(row.data[self.forces_property])
        except (KeyError, AttributeError):
            forces = np.zeros((len(atoms), 3))
        default_type = torch.get_default_dtype()

        # pylint: disable=E1102
        graph_data = {
            "nodes": torch.tensor(atoms.get_atomic_numbers()),
            "nodes_xyz": torch.tensor(atoms.get_positions(), dtype=default_type),
            "num_nodes": torch.tensor(len(atoms.get_atomic_numbers())),
            "edges": torch.tensor(edges),
            "edges_displacement": torch.tensor(edges_displacement, dtype=default_type),
            "cell": torch.tensor(np.array(atoms.get_cell()), dtype=default_type),
            "num_edges": torch.tensor(edges.shape[0]),
            "energy": torch.tensor(energy, dtype=default_type),
            "forces": torch.tensor(forces, dtype=default_type),
        }

        return graph_data

    def get_edges(self, atoms):
        # Compute distance matrix
        pos = atoms.get_positions()
        dist_mat = scipy.spatial.distance_matrix(pos, pos)

        # Build array with edges and edge features (distances)
        valid_indices_bool = dist_mat < self.cutoff
        np.fill_diagonal(valid_indices_bool, False)  # Remove self-loops
        edges = np.argwhere(valid_indices_bool)  # num_edges x 2
        edges_displacement = np.zeros((edges.shape[0], 3))

        return edges, edges_displacement
