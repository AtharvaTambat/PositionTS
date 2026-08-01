"""Main-results generation of transition states for the rgdy test split, using the rgdy-trained checkpoint."""

from typing import List
import time
import os
import numpy as np
import time
import os
import numpy as np
import torch
import pickle
import argparse
from uuid import uuid4

from torch.utils.data import DataLoader

from reactot.trainer.pl_trainer import DDPMModule, SBModule
from reactot.dataset.transition1x import ProcessedTS1x
# from reactot.analyze.rmsd import batch_rmsd, batch_rmsd_n_dmae
# from reactot.analyze.geomopt import calc_deltaE, compute_efh
from reactot.evaluate.utils import (
    set_new_schedule,
    inplaint_batch,
    sample_reactot_batch,
    batch_ts_deltaE,
)
from reactot.utils.sampling_tools import write_single_xyz
# from reactot.trainer.train_rpsb_ts1x import OPT

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5

def write_temp_xyz(
            fragments_nodes, out_samples, idx, prefix="gen", localpath="tmp", ex_ind=0
):
    """Write <localpath>/<prefix>_<ex_ind+j>.xyz per molecule; ex_ind carries the cross-batch offset."""
    for ii in idx:
        start_ind, end_ind = 0, 0
        for jj, natoms in enumerate(fragments_nodes[0]):
            _jj = jj + ex_ind
            print(_jj, flush=True)
            xyzfile = f"{localpath}/{prefix}_{_jj}.xyz"
            end_ind += natoms.item()
            write_single_xyz(
                xyzfile,
                natoms.item(),
                out=out_samples[ii][start_ind:end_ind],
            )
            start_ind = end_ind

class OPT:
    """Solver settings consumed by the Schrodinger-bridge sampler."""
    def __init__(
        self,
        solver,
        method,
    ):
        self.solver = solver
        self.method = method
        self.atol = 1e-2
        self.rtol = 1e-2
    
opt = OPT(solver="ddpm", method="midpoint")

config = dict(
    timesteps=150,
    bz=16,
    resamplings=5,
    jump_length=5,
    repeats=1,
    max_batch=-1,
    shuffle=False,
    single_frag_only=False,
)

print("loading ddpm trainer...")
device = torch.device("cuda")
# Directory holding the trained checkpoints, including the wandb project
# subdirectory: <--checkpoint-root>/<--project> as passed to trainer/train.py,
# e.g. "../trainer/checkpoint/ReactOT_rgd1/".
tspath = ""
checkpoints = {
    "repro": f"{tspath}/Adam-AF-False-LRSNone-EENone-PT-TS1x-Edge-SinEmTrue-L9-Unit128-NF1.0-AGGmean-SubL1-rmHFalse-BZ64-Norm1.0_1.0_1.0-emaFalse-POTrue-Decay0-SFTrue/ddpm-epoch=4379-val-totloss=602.40.ckpt",
    "chiral": f"{tspath}/5edcbc9baced/ddpm-epoch=4159-val-totloss=585.78.ckpt",
    "leftnet_legacy": f"{tspath}/leftnet-78c7590798bc/ddpm-epoch=1059-val-totloss=648.90.ckpt",
    "leftnet_2074": f"{tspath}-All/leftnet-8-70b75beeaac1/ddpm-epoch=2074-val-totloss=531.18.ckpt",
    "leftnet_actual_test_best_rgdy": f"{tspath}/leftnet-ts_guess_NEBCI-xtb-ema-d2d3f73e48cc/sb-epoch=1586-val_ep_scaled_err=0.0550.ckpt"
}
ddpm_trainer = SBModule.load_from_checkpoint(
    checkpoint_path=checkpoints["leftnet_actual_test_best_rgdy"],
    map_location=device,
    strict=False,
)

ddpm_trainer.nfe = 100
ddpm_trainer.ddpm.opt = opt
ddpm_trainer = ddpm_trainer.to(device)
# ddpm_trainer = set_new_schedule(
#     ddpm_trainer,
#     timesteps=config["timesteps"],
#     noise_schedule="polynomial_2",
# )

print("loading dataset...")
dataset = ProcessedTS1x(
    # Test split for this dataset, e.g. "../data/rgdy_splits/test_split.pkl"
    pkl_path="",
    center=True,
    pad_fragments=0,
    device="cuda",
    zero_charge=False,
    remove_h=False,
    single_frag_only=config["single_frag_only"],
    swapping_react_prod=False,
    use_by_ind=True,
)
loader = DataLoader(
    dataset,
    batch_size=config["bz"],
    shuffle=config["shuffle"],
    num_workers=0,
    collate_fn=dataset.collate_fn
)
_id = uuid4()
localpath = "tmp_rgdy/"
os.makedirs(localpath, exist_ok=True)

print("evaluating...")
for num_repeat in range(config["repeats"]):
    print("num_repeat: ", num_repeat)
    _rmsds, _deltaEs, _dmaes = [], [], []
    for ii, batch in enumerate(loader):
        print("batch_idx: ", ii)
        time_start = time.time()
        # if ii == config["max_batch"]:
        #     break
        out_samples, xh_fixed, fragments_nodes = sample_reactot_batch(
            batch,
            ddpm_trainer,
            resamplings=config["resamplings"],
            jump_length=config["jump_length"],
        )
        write_temp_xyz(fragments_nodes, [out_samples], idx=[0], prefix="entry", localpath=localpath, ex_ind=(ii*config["bz"]))
        