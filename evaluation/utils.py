"""Batch sampling and energy helpers for the react-ot models.

Dead copy of baselines/react-ot/reactot/evaluate/utils.py: nothing here imports it, and
its `reactot.*` imports do not resolve from this directory.
"""
from typing import List
import torch
import time

from reactot.trainer.pl_trainer import DDPMModule, SBModule
from reactot.diffusion._schedule import DiffSchedule, PredefinedNoiseSchedule
from reactot.diffusion._normalizer import FEATURE_MAPPING
from reactot.analyze.geomopt import calc_deltaE, compute_efh

EV2KCALMOL = 23.06
AU2KCALMOL = 627.5
device = torch.device("cuda")


def set_new_schedule(
    ddpm_trainer: DDPMModule,
    timesteps: int = 250,
    device: torch.device = torch.device("cuda"),
    noise_schedule: str = "polynomial_2"
) -> DDPMModule:
    """Swap in a fresh noise schedule with the given number of timesteps and move the module to device."""
    precision: float = 1e-5

    gamma_module = PredefinedNoiseSchedule(
        noise_schedule=noise_schedule,
        timesteps=timesteps,
        precision=precision,
    )
    schedule = DiffSchedule(
        gamma_module=gamma_module,
        norm_values=ddpm_trainer.ddpm.norm_values
    )
    ddpm_trainer.ddpm.schedule = schedule
    ddpm_trainer.ddpm.T = timesteps
    return ddpm_trainer.to(device)


def inplaint_batch(
    batch: List,
    ddpm_trainer: DDPMModule,
    resamplings: int = 1,
    jump_length: int = 1,
    frag_fixed: List = [0, 2],
):
    """Inpaint one batch with the fragments in frag_fixed held at their input geometry.

    Returns the sampled features, the fixed input features and the per-fragment atom counts.
    """
    representations, conditions = batch
    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations
    ]
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [
        repre["size"] for repre in representations
    ]
    out_samples, _ = ddpm_trainer.ddpm.inpaint(
        n_samples=n_samples,
        fragments_nodes=fragments_nodes,
        conditions=conditions,
        return_frames=1,
        resamplings=resamplings,
        jump_length=jump_length,
        timesteps=None,
        xh_fixed=xh_fixed,
        frag_fixed=frag_fixed,
    )
    return out_samples[0], xh_fixed, fragments_nodes
    

def sample_reactot_batch(
    batch: List,
    ddpm_trainer: SBModule,
    resamplings: int = 1,
    jump_length: int = 1,
    frag_fixed: List = [0, 2],
):
    """Sample one batch with the Schrodinger-bridge model, returning predicted and target features.

    resamplings, jump_length and frag_fixed are accepted for signature compatibility with
    inplaint_batch and are not used.
    """
    representations, conditions = batch
    xh_fixed = [
        torch.cat(
            [repre[feature_type] for feature_type in FEATURE_MAPPING],
            dim=1,
        )
        for repre in representations
    ]
    n_samples = representations[0]["size"].size(0)
    fragments_nodes = [
        repre["size"] for repre in representations
    ]
    x0, x1, cond, x0_size, x0_other = ddpm_trainer.ddpm.sample_batch(
        representations, conditions, return_timesteps=False, training=False)

    with torch.no_grad():
        xs, pred_x0 = ddpm_trainer.ddpm.sample(
            x1, representations, conditions, nfe=ddpm_trainer.nfe, ot_ode=ddpm_trainer.ot_ode)
        
    x0_pred = xs[:, 0, ...]

    target_xh = torch.cat([x0.cpu(),      x0_other.cpu()], dim=1)
    pred_xh   = torch.cat([x0_pred.cpu(), x0_other.cpu()], dim=1)
    
    return pred_xh, target_xh, fragments_nodes

def batch_ts_deltaE(bz, xc="wb97x", basis="631g*", d3=False, localpath="tmp"):
    """E(gen_<i>_ts) - E(sample_<i>_ts) in kcal/mol, for each of the bz pairs under localpath."""
    deltaEs = []
    for ii in range(bz):
        s_time = time.time()
        deltaEs.append(
            calc_deltaE(
                f"{localpath}/sample_{ii}_ts.xyz",
                f"{localpath}/gen_{ii}_ts.xyz",
                xc=xc,
                basis=basis,
                d3=d3,
            ) * EV2KCALMOL
        )
        print("time cost: ", time.time() - s_time)
        print("----")
    return deltaEs


def batch_E(bz, prefix="gen"):
    """Total energies in kcal/mol of tmp/<prefix>_<i>_ts.xyz for i below bz."""
    Es = []
    for ii in range(bz):
        mf, _, _ = compute_efh(
            f"tmp/{prefix}_{ii}_ts.xyz",
            f=False, hess=False, return_metrics=False)
        Es.append(mf.e_tot * AU2KCALMOL)
    return Es


def samples_to_pos_charge(out_samples, fragments_nodes):
    """Split a stacked batch into per-molecule coordinates keyed by fragment, atomic numbers and atom counts."""
    x_r = torch.tensor_split(
        out_samples[0],
        torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    x_ts = torch.tensor_split(
        out_samples[1],
        torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    x_p = torch.tensor_split(
        out_samples[2],
        torch.cumsum(fragments_nodes[0], dim=0).to("cpu")[:-1]
    )
    pos = {
        "reactant": [_x[:, :3].cpu().numpy() for _x in x_r],
        "transition_state": [_x[:, :3].cpu().numpy() for _x in x_ts],
        "product": [_x[:, :3].cpu().numpy() for _x in x_p],
    }
    z = [_x[:, -1].long().cpu().numpy() for _x in x_r]
    natoms = [f.cpu().item() for f in fragments_nodes[0]]
    return pos, z, natoms
