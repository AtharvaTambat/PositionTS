import copy
import numpy as np
import torch
import pickle
from oa_reactdiff.dataset.base_dataset import BaseDataset, ATOM_MAPPING
from oa_reactdiff.dataset.toxify import create_permutation_matrix

n_element = len(list(ATOM_MAPPING.keys()))
FRAG_MAPPING = {
    "reactant": "product",
    "transition_state": "transition_state",
    "product": "reactant",
}

# Fragment order produced by collate_fn is [reactant, transition_state, product].
PERMUTED_FRAGMENTS = {
    "none": (),
    "product": (2,),
    "ts": (1,),
    "product+ts": (1, 2),
}


def reflect_z(x):
    x = np.array(x)
    x[:, -1] = -x[:, -1]
    return x


class ProcessedTS1x_woTest(BaseDataset):
    def __init__(
        self,
        npz_path,
        center=True,
        pad_fragments=0,
        device="cpu",
        zero_charge=False,
        remove_h=False,
        single_frag_only=True,
        swapping_react_prod=False,
        append_frag=False,
        reflection=False,
        use_by_ind=False,
        only_ts=False,
        confidence_model=False,
        position_key="positions",
        ediff=None,
        **kwargs,
    ):
        super().__init__(
            npz_path=npz_path,
            center=center,
            device=device,
            zero_charge=zero_charge,
            remove_h=remove_h,
        )
        if confidence_model:
            use_by_ind = False
        if remove_h:
            print("remove_h is ignored because it is not reasonble for TS.")
        if single_frag_only:
            single_frag_inds = np.where(
                np.array(self.raw_dataset["single_fragment"]) == 1
            )[0]
        else:
            single_frag_inds = np.array(range(len(self.raw_dataset["single_fragment"])))
        if use_by_ind:
            use_inds = self.raw_dataset["use_ind"]
        else:
            use_inds = range(len(self.raw_dataset["single_fragment"]))
        single_frag_inds = list(set(single_frag_inds).intersection(set(use_inds)))

        data_duplicated = copy.deepcopy(self.raw_dataset)
        for k, mapped_k in FRAG_MAPPING.items():
            for v, val in data_duplicated[k].items():
                self.raw_dataset[k][v] = [val[ii] for ii in single_frag_inds]
                if swapping_react_prod:
                    mapped_val = data_duplicated[mapped_k][v]
                    self.raw_dataset[k][v] += [
                        mapped_val[ii] for ii in single_frag_inds
                    ]
        if reflection:
            for k, mapped_k in FRAG_MAPPING.items():
                for v, val in self.raw_dataset[k].items():
                    if v in ["wB97x_6-31G(d).forces", position_key]:
                        self.raw_dataset[k][v] += [reflect_z(_val) for _val in val]
                    else:
                        self.raw_dataset[k][v] += val

        self.reactant = self.raw_dataset["reactant"]
        self.transition_state = self.raw_dataset["transition_state"]
        self.product = self.raw_dataset["product"]

        self.n_fragments = pad_fragments + 3
        self.device = torch.device(device)
        n_samples = len(self.reactant["charges"])
        self.n_samples = len(self.reactant["charges"])

        self.data = {}
        repeat = 2 if swapping_react_prod else 1
        if confidence_model:
            self.data["target"] = torch.tensor(
                self.raw_dataset["target"] * repeat
            ).unsqueeze(1)
            self.data["rmsd"] = torch.tensor(
                self.raw_dataset["rmsd"] * repeat
            ).unsqueeze(1)
        if ediff is not None:
            self.data["ediff"] = torch.tensor(
                self.raw_dataset[ediff]["ediff"] * repeat
            ).unsqueeze(1)
        if not only_ts:
            if not append_frag:
                self.process_molecules(
                    "reactant", n_samples, idx=0, position_key=position_key
                )
                self.process_molecules("transition_state", n_samples, idx=1)
                self.process_molecules(
                    "product", n_samples, idx=2, position_key=position_key
                )
            else:
                self.process_molecules(
                    "reactant",
                    n_samples,
                    idx=0,
                    append_charge=0,
                    position_key=position_key,
                )
                self.process_molecules(
                    "transition_state", n_samples, idx=1, append_charge=1
                )
                self.process_molecules(
                    "product",
                    n_samples,
                    idx=2,
                    append_charge=0,
                    position_key=position_key,
                )

            for idx in range(pad_fragments):
                self.patch_dummy_molecules(idx + 3)
        else:
            if not append_frag:
                self.process_molecules("transition_state", n_samples, idx=0)
            else:
                self.process_molecules(
                    "transition_state", n_samples, idx=0, append_charge=1
                )
            # for idx in range(2):
            #     self.patch_dummy_molecules(idx + 1)

        self.data["condition"] = [
            torch.zeros(
                size=(1, 1),
                dtype=torch.int64,
                device=self.device,
            )
            for _ in range(self.n_samples)
        ]

class ProcessedTS1x():
    def __init__(
        self,
        pkl_path,
        center=True,
        pad_fragments=0,
        device="cpu",
        permute="none",
        **kwargs,
    ):

        # Load data from pickle file
        with open(pkl_path, "rb") as f:
            self.raw_dataset = pickle.load(f)

        self.n_fragments = pad_fragments + 3
        self.device = torch.device(device)
        self.n_samples = len(self.raw_dataset)
        if permute not in PERMUTED_FRAGMENTS:
            raise KeyError(
                f"permute must be one of {sorted(PERMUTED_FRAGMENTS)}, got {permute!r}"
            )
        self.permuted_fragments = PERMUTED_FRAGMENTS[permute]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        batch = self.raw_dataset[idx]
        for x in batch:
            batch[x] = torch.tensor(batch[x], device=self.device)
        return batch
    
    # ============================================================================
    # FLAW (this work, paper Sec. 4.2(I), Table 3)
    # Atom-mapping probe. Permuting product atom order leaves the physical
    # system unchanged, so a permutation-equivariant model should be unaffected.
    # Most baselines degrade by 2-4x in RMSD, which is the evidence that they
    # silently rely on atom mapping being available at test time.
    # ============================================================================
    def permute_fragments(self, out):
        """Relabel atoms of the selected fragments with one shared permutation."""
        if not self.permuted_fragments:
            return
        pms = create_permutation_matrix(out[self.permuted_fragments[0]]["size"])
        for idx in self.permuted_fragments:
            out[idx]["pos"] = pms @ out[idx]["pos"]
            out[idx]["one_hot"] = (pms.float() @ out[idx]["one_hot"].float()).long()
            out[idx]["charge"] = (pms.float() @ out[idx]["charge"].float()).int()

    def collate_fn(self, batch):
        sizes = []
        bs = len(batch)
        for k in batch[0].keys():
            if "size" in k:
                sizes.append(int(k.split("_")[-1]))
        n_fragment = len(sizes)
        out = [{} for _ in range(n_fragment)]
        res = {}
        for prop in batch[0].keys():
            if prop not in ["condition", "target", "rmsd", "ediff"]:
                idx = int(prop.split("_")[-1])
                _prop = prop.replace(f"_{idx}", "")
            if "size" in prop:
                out[idx][_prop] = torch.tensor(
                    [x[prop] for x in batch],
                    device=batch[0][prop].device,
                )
            elif "mask" in prop:
                out[idx][_prop] = torch.cat(
                    [
                        i * torch.ones(len(x[prop]), device=x[prop].device).long()
                        for i, x in enumerate(batch)
                    ],
                    dim=0,
                )
            elif prop in ["condition", "target", "rmsd", "ediff"]:
                res[prop] = torch.cat([x[prop] for x in batch], dim=0)
            elif _prop in ["energy", "atomization_energy"]:
                out[idx][_prop] = torch.cat([x[prop].reshape(1) for x in batch], dim=0)
            else:
                out[idx][_prop] = torch.cat([x[prop] for x in batch], dim=0)
                # print(out[idx][_prop].device)
        self.permute_fragments(out)
        if len(list(res.keys())) == 1:
            return out, res["condition"]
        return out, res