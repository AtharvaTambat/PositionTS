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

from oa_reactdiff.trainer.pl_trainer import DDPMModule
from oa_reactdiff.dataset.transition1x import ProcessedTS1x
# from oa_reactdiff.analyze.rmsd import batch_rmsd
# from oa_reactdiff.analyze.geomopt import calc_deltaE, compute_efh
from oa_reactdiff.evaluate.utils import (
    set_new_schedule,
    inplaint_batch,
    batch_ts_deltaE,
)
from oa_reactdiff.utils.sampling_tools import write_single_xyz

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5

def write_temp_xyz(
            fragments_nodes, out_samples, idx, prefix="gen", localpath="tmp", ex_ind=0
):
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

print("loading ddpm trainer...", flush=True)
device = torch.device("cuda")
# tspath = "/home/ubuntu/efs/2Dto3D_ReactGen/oa_reactdiff/trainer/ckpt/TSDiffusion-TS1x"
tspath = "/home/rbs-ml/205030033/ankit/OAReactDiff/oa_reactdiff/trainer/checkpoint" 
checkpoints = {
    "leftnet_1301": f"{tspath}/OAReactDiff/leftnet-0-b670188b2cee/ddpm-epoch=1301-val-totloss=542.96.ckpt",
    "leftnet_1999": f"{tspath}/OAReactDiff/leftnet-0-31293be7327e/ddpm-epoch=1999-val-totloss=647.19.ckpt",
    "leftnet_711": f"{tspath}/OAReactDiff/leftnet-0-97458639218ats1x_manual_split_test_subset_100_T_5000_num_layers_3_sigma_0_100/ddpm-epoch=711-val-totloss=552.38.ckpt",
    "leftnet_at_final": f"/home/rbs-ml/205030033/ankit/OAReactDiff/oa_reactdiff/checkpoints/pretrained-ts1x-epoch2000-bz16-AT-checked.ckpt",
    "leftnet_atag_best": f"{tspath}/OAReactDiff/leftnet-0-e1782ceedd16ts1x_manual_split_test_subset_100_T_5000_num_layers_3_sigma_0_100/ddpm-epoch=1697-val-totloss=529.11.ckpt",
    "leftnet_atag_test_best": f"{tspath}/OAReactDiff/leftnet-0-41d30768408dts1x_manual_split_test_subset_100_T_5000_num_layers_3_sigma_0_100/ddpm-epoch=1371-val-totloss=537.06.ckpt",
    "leftnet_atag_test_last": "/home/rbs-ml/205030033/ankit/OAReactDiff/oa_reactdiff/checkpoints/pretrained-ts1x-epoch2000-bz16-nlayer3-tvt-ATAG.ckpt",
    "leftnet_atag_test_best2": f"{tspath}/OAReactDiff/leftnet-0-392f3f25df75ts1x_manual_split_test_subset_100_T_5000_num_layers_3_sigma_0_100/ddpm-epoch=1697-val-totloss=528.16.ckpt",
    "leftnet_actual_test_best_rgdx": f"{tspath}/OAReactDiff_rgdx_limit200/leftnet-0-d55fe2b9b2d0/ddpm-epoch=1724-val-totloss=625.38.ckpt",
}
ddpm_trainer = DDPMModule.load_from_checkpoint(
    checkpoint_path=checkpoints["leftnet_actual_test_best_rgdx"],
    map_location=device,
)
ddpm_trainer = set_new_schedule(
    ddpm_trainer,
    timesteps=config["timesteps"],
    noise_schedule="polynomial_2",
)

print("loading dataset...", flush=True)
dataset = ProcessedTS1x(
    pkl_path="/home/rbs-ml/205030033/ankit/ts_project/OAReactDiff/oa_reactdiff/data/rgdx_splits/test_split.pkl",
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
    collate_fn=dataset.collate_fn,
)
_id = uuid4()
localpath = "tmp_rgdx/"
os.makedirs(localpath)
for num_repeat in range(config["repeats"]):
    print("num_repeat: ", num_repeat, flush=True)
    for ii, batch in enumerate(loader):
        print("batch_idx: ", ii, flush=True)
        time_start = time.time()
        # if ii == config["max_batch"]:
        #     break
        out_samples, xh_fixed, fragments_nodes = inplaint_batch(
            batch,
            ddpm_trainer,
            resamplings=config["resamplings"],
            jump_length=config["jump_length"],
        )
        write_temp_xyz(fragments_nodes, out_samples, idx=[1], prefix="entry", localpath=localpath, ex_ind=(ii*config["bz"]))
        