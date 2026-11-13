import sys
import numpy as np
import torch
import pytorch_lightning as pl
from nehod import flows

# Samples the 27-target flow at fiducial TNG parameters, conditioned on the M200c of each
# disk-selected TNG central so the flow and TNG samples share a halo mass distribution.
# usage: python generate_tng.py tng50|tng100

flow_ckpt = "Model/gse-prob-flows-model_2/lightning_logs/checkpoints/epoch=991-val_loss=10.1514.ckpt"
cat_path = "../TNGdata/analysis/np_arrays"
out_path = "../TNGdata/analysis/generated_data"

n_per_gal = 500
mfof_min, mfof_max = 5e11, 2.5e12   #DREAMS.get_MW_idx selection

#Planck 2015 cosmology used by TNG, fiducial TNG feedback, no contamination
omega_m, sigma_8 = 0.3089, 0.8159
tng_params = [omega_m, sigma_8, np.log10(3.6), np.log10(7.4), np.log10(0.1), 0]

target_names = ['mm_time', 'gse_time', 'gse_ratio', 'ngse', 'halo_f', 'halo_beta',
    'halo_tot_smass', 'halo_sat_smass', 'halo_bmass', 'halo_bmass_frac', 'halo_sfr',
    'bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'bulge_dm_frac', 'bulge_vdisp',
    'disk_vcirc', 'disk_smass', 'disk_size', 'disk_height',
    'halo_central_smass', 'halo_break', 'halo_slope_in', 'halo_slope_out', 'halo_q_in', 'halo_q_out']

def main():

    name = sys.argv[1]
    pl.seed_everything(13)

    cat = np.load(f"{cat_path}/catalog_{name}.npz")
    mfof = np.power(10, cat['log_mfof'])
    sel = cat['has_disk'] & (mfof > mfof_min) & (mfof < mfof_max)
    log_m200c = cat['log_m200c'][sel]
    print(name, 'disk galaxies in mass range:', np.sum(sel))

    model = flows.NPE.load_from_checkpoint(flow_ckpt, map_location='cpu')
    model.eval()
    norm_dict = model.norm_dict

    gal_idx = np.repeat(np.arange(len(log_m200c)), n_per_gal)
    context = np.empty((len(gal_idx), 7))
    context[:,0] = log_m200c[gal_idx]
    context[:,1:] = tng_params

    context_norm = (context - norm_dict['cond_mean']) / norm_dict['cond_std']
    context_norm = torch.tensor(context_norm, dtype=torch.float32)

    samples = []
    with torch.no_grad():
        for batch in torch.split(context_norm, 100000):
            samples.append(model.flow(model(batch)).sample().numpy())
    samples = np.concatenate(samples) * norm_dict['target_std'] + norm_dict['target_mean']

    np.savez(f"{out_path}/flow_{name}_TNG",
             samples=samples, target_names=np.array(target_names), context=context,
             gal_idx=gal_idx, sub_id=cat['sub_id'][sel], log_m200c=log_m200c)

    return

if __name__=='__main__':
    main()
