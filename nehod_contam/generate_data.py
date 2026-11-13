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
from nehod import flows, eval

def get_context(name, mi, ma, n_samples):


    if name == 'CV':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations
        context = np.ones((n_samples, 7))*np.array([12, .314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1), 0]) #CV cdm
        #context[:,1:] = cv_context

    if name == 'best':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations
        context = np.ones((n_samples, 7))*np.array([12, .314, .834, np.log10(5.75), np.log10(8.91), np.log10(0.1), 0]) #CV cdm
        #context[:,1:] = cv_context

    elif name == 'low':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations
        cv_context = np.ones((n_samples, 5))*np.array([.314, .834, np.log10(3.6/4+0.2), np.log10(7.4), np.log10(0.1)]) #CV cdm
        context[:,1:] = cv_context

    elif name =='SB':
        context = np.random.uniform(mi, ma, size=(n_samples, len(mi))) #astro variations
        context[:,-1] = np.zeros(n_samples)

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

    elif name=='cont':
        sb_context = np.random.uniform(mi, ma, size=(n_samples, len(mi)))
        context = np.ones((n_samples, 7))*np.array([12,.314, .834, np.log10(3.6), np.log10(7.4), np.log10(0.1), 0]) 
        context[:,6] = sb_context[:,6]

    return context

def get_ckpt(trial):

    files = os.listdir(f"Model/gse-prob-flows-model_{trial}/lightning_logs/checkpoints/")
    files = [f for f in files if 'epoch' in f]
    losses = [float(f.split("=")[2].split('.')[0]) for f in files]

    idx = np.argmin(losses)
    
    return files[idx]


def main():
    """Generate samples from the Appendix E contamination emulator.

    NOT YET RESTORED (see ISSUES.md). The research version of this script was later
    overwritten by a satellite/VDM generator; that code has been removed from the release.
    The paper needs:
      * Fig E2: 1e5 samples with get_context('cont', ...): M_TOT = 1e12 Msun, fiducial
        cosmological/astrophysical parameters, contamination uniform in [0, 0.02]
        -> generated_samples/100k_CDM_cont.npy (read by contam_comparison.py)
      * Fig E3: samples with contamination fixed to 0 (get_context('SB', ...)), from which
        the GSE subset is selected (read by gse_comparison.py)
    using the 27-target model(s) in Model/gse-prob-flows-model_{2,3,4} (paper models) or a
    retrained model. The flow-sampling loop in nehod/generate_data.py can be used as a template.
    """
    raise NotImplementedError("Appendix E sample generation must be restored; see the docstring and ISSUES.md")


if __name__=='__main__':
    main()
