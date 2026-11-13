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

def get_context(name, mi, ma, n_samples):


    if name == 'CV':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations
        cv_context = np.ones((n_samples, 5))*np.array([.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1)]) #CV cdm
        context[:,1:] = cv_context

    elif name == 'low':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations
        cv_context = np.ones((n_samples, 5))*np.array([.314, .834, np.log10(3.6/4+0.2), np.log10(7.4), np.log10(0.1)]) #CV cdm
        context[:,1:] = cv_context

    elif name =='SB':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations

    elif name=='Om':
        sb_context = np.random.uniform(mi, ma, size=(n_samples, len(mi)))
        context = np.ones((n_samples, 6))*np.array([12,.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1)]) 
        context[:,0] = sb_context[:,0]
        context[:,1] = sb_context[:,1]

    elif name=='s8':
        sb_context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) 
        context = np.ones((n_samples, 6))*np.array([12,.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1)]) 
        context[:,0] = sb_context[:,0]
        context[:,2] = sb_context[:,2]

    elif name=='SN1':
        sb_context = np.random.uniform(mi, ma, size=(n_samples, len(mi)))
        context = np.ones((n_samples, 6))*np.array([12,.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1)]) 
        context[:,0] = sb_context[:,0]
        context[:,3] = sb_context[:,3]

    elif name=='SN2':
        sb_context = np.random.uniform(mi, ma, size=(n_samples, len(mi)))
        context = np.ones((n_samples, 6))*np.array([12,.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1)]) 
        context[:,0] = sb_context[:,0]
        context[:,4] = sb_context[:,4]

    elif name=='AGN':
        sb_context = np.random.uniform(mi, ma, size=(n_samples, len(mi)))
        context = np.ones((n_samples, 6))*np.array([12,.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1)]) 
        context[:,0] = sb_context[:,0]
        context[:,5] = sb_context[:,5]

    return context

def get_ckpt(trial):

    files = os.listdir(f"Model/gse-prob-flows-model_{trial}/lightning_logs/checkpoints/")
    files = [f for f in files if 'epoch' in f]
    losses = [float(f.split("=")[2].split('.')[0]) for f in files]

    idx = np.argmin(losses)
    
    return files[idx]


def main():

    trials = [38,48,50,55,64,80,97,99,103,108]
    all_data = []

    # Release: the research code toggled `data_type` by hand. Pass it on the command line instead:
    #   SB  -> 1000k_CDM_SB_gse_prob.npy (main 1e6 sample; Figs 5, A1-A4, A7, A8, B2, B3)
    #   CV / low -> fixed-physics variants;  Om, s8, SN1, SN2, AGN -> one-parameter variations (Fig 4)
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'SB'
    per_model_types = ['Om', 's8', 'SN1', 'SN2', 'AGN']

    for i in range(10):
        # Release: the research code re-seeded every model with the same seed (10), so all ten
        # ensemble members received identical contexts and base-distribution draws.
        pl.seed_everything(10 + i)
        n_samples = 100000

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        file = get_ckpt(trials[i])
        print(file)

        flow_model = flows.NPE.load_from_checkpoint(f"Model/gse-prob-flows-model_{trials[i]}/lightning_logs/checkpoints/{file}", map_location=device)

        # make sure that the models are in evaluation mode
        #flow_model = flows.NPE.load_from_checkpoint("Model/gse-prob-flows-model_108/lightning_logs/checkpoints/epoch=1166-val_loss=9.7829.ckpt", map_location=device)
        #flow_model = flows.NPE.load_from_checkpoint("Model/gse-prob-flows-model_34/lightning_logs/checkpoints/epoch=316-val_loss=12.8417.ckpt", map_location=device)

        flow_model.eval()
        flow_model.to(device)

        # load the normalization statistics
        flow_norm_dict = flow_model.norm_dict

        context_min = np.array(
            [11.7, 0.274, 0.780, np.log10(0.9), np.log10(3.7), np.log10(0.025)]) #cdm
        context_max = np.array(
            [12.3, 0.354, 0.888, np.log10(14.4), np.log10(14.8), np.log10(0.4)]) #cdm

        context = get_context(data_type, context_min, context_max, n_samples)

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

        if data_type in per_model_types:  # Fig 4 reads one file per ensemble member
            np.save(f'generated_samples/{n_samples//1000}k_CDM_{data_type}_{i}_gse_prob', diffusion_context)

    all_data = np.array(all_data)
    print(all_data.shape)
    np.save(f'generated_samples/{len(all_data)//1000}k_CDM_{data_type}_gse_prob', all_data)

    return


if __name__=='__main__':
    main()
