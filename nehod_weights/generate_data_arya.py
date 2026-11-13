import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import zuko
from torch.utils.data import DataLoader, TensorDataset

import datasets
from nehod import flows, vdm, eval
from nehod.diffusion_utils import alpha, sigma2, variance_preserving_map

def get_context(mi, ma, nsamples_per_bin):

    nspb = nsamples_per_bin
    nsamples_tot = nsamples_per_bin * 30**3

    all_sn1 = np.linspace(mi[2], ma[2], 30)
    all_sn2 = np.linspace(mi[3], ma[3], 30)
    all_agn = np.linspace(mi[4], ma[4], 30)

    context = np.zeros((nsamples_tot,5))
    context[:,0] = np.random.uniform(mi[0], ma[0], size=nsamples_tot)
    context[:,1] = np.random.uniform(mi[1], ma[1], size=nsamples_tot)
    for sn1 in range(30):
        for sn2 in range(30):
            for agn in range(30):
                idx = sn1*900 + sn2*30 + agn
                context[idx*nspb:(idx+1)*nspb,2] = all_sn1[sn1]
                context[idx*nspb:(idx+1)*nspb,3] = all_sn2[sn2]
                context[idx*nspb:(idx+1)*nspb,4] = all_agn[agn]
               
    return context 
                

def get_ckpt(trial):

    files = os.listdir(f"Model/master-cdm-flows-model_{trial}/lightning_logs/checkpoints/")
    files = [f for f in files if 'epoch' in f]
    losses = [float(f.split("=")[2].split('.')[0]) for f in files]

    idx = np.argmin(losses)
    
    return files[idx]


def main():

    #trials = [9,13,14,15,16]
    #trials = [34,32,30,25,22,18,14,5]
    trials = [34,25,22,18,5]
    all_data = []

    # Release: 200 samples per model per bin x 5 models = 1000 per bin, as used for the paper's
    # generated_samples_arya/1000_arya_cdm.npy (and assumed by calc_weights_arya.py).
    nsamples_per_bin = 200 #per trial
    data_type = 'SB'

    context_min = np.array(
        [0.274, 0.780, np.log10(0.9), np.log10(3.7), np.log10(0.025)]) #cdm
    context_max = np.array(
        [0.354, 0.888, np.log10(14.4), np.log10(14.8), np.log10(0.4)]) #cdm

    np.random.seed(13)  # Release: seed the Omega_m / sigma_8 draws (previously unseeded)
    context = get_context(context_min, context_max, nsamples_per_bin)

    for i in range(len(trials)):
        pl.seed_everything(13)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        file = get_ckpt(trials[i])
        print(file)

        flow_model = flows.NPE.load_from_checkpoint(f"Model/master-cdm-flows-model_{trials[i]}/lightning_logs/checkpoints/{file}", map_location=device)

        flow_model.eval()
        flow_model.to(device)

        # load the normalization statistics
        flow_norm_dict = flow_model.norm_dict

        context_norm = (context - flow_norm_dict['cond_mean']) / flow_norm_dict['cond_std']
        context_norm = torch.tensor(context_norm, dtype=torch.float32)
        context_norm = context_norm.to(device)

        with torch.no_grad():
            flow_context = flow_model(context_norm)
            targets = flow_model.flow(flow_context).sample()
            norm_targets = targets.cpu().numpy()

        denorm_targets = norm_targets * flow_norm_dict['target_std'] + flow_norm_dict['target_mean']
        diffusion_context = np.hstack([context, denorm_targets])
        all_data.extend(diffusion_context)

        #np.save(f'generated_samples/{n_samples//1000}k_CDM_{data_type}_{i}_gse_prob', diffusion_context)

    all_data = np.array(all_data)
    print(all_data.shape)
    np.save(f'generated_samples_arya/{int(nsamples_per_bin*len(trials))}_arya_cdm', all_data)

    return


if __name__=='__main__':
    main()
