import numpy as np
import pandas as pd

def main():

    name = 'gse_prob'

    header = 'nsat,total_mass,stellar_mass,omega_m,sigma_8,log_sn1,log_sn2,log_agn1,mm_time,gse_time,gse_ratio,ngse,halo_f,halo_beta,'
    header += 'halo_tot_smass,halo_sat_smass,halo_bmass,halo_bmass_frac,halo_sfr,'
    header += 'bulge_reff,bulge_mass,bulge_smass,bulge_bhmass,bulge_dm_frac,bulge_vdisp,'
    header += 'disk_vcirc,disk_smass,disk_size,disk_height,'
    header += 'halo_central_smass,halo_break,halo_slope_in,halo_slope_out,halo_q_in,halo_q_out,'
    header += 'contam'

    data = np.load(f'{name}.npz')
    cond = pd.read_csv(f'{name}_cond.csv')

    data_norm = data['features']
    mask = data['mask']
    cond_norm = cond.to_numpy()

    train_data = 0.8
    valid_data = 0.2
    test_data = 0.0

    nsims = len(cond)
    sim_idx = np.arange(nsims)
    np.random.default_rng(13).shuffle(sim_idx)  # Release: seeded split

    data_norm_train = data_norm[sim_idx][:int(nsims*train_data)]
    cond_norm_train = cond_norm[sim_idx][:int(nsims*train_data)]
    mask_train = mask[sim_idx][:int(nsims*train_data)]

    np.savez(f'train_{name}', features=data_norm_train, mask=mask_train)
    np.savetxt(f"train_{name}_cond.csv", cond_norm_train, delimiter=',', header=header, comments='')

    data_norm_valid = data_norm[sim_idx][int(nsims*train_data):int(nsims*train_data)+int(nsims*valid_data)]
    cond_norm_valid = cond_norm[sim_idx][int(nsims*train_data):int(nsims*train_data)+int(nsims*valid_data)]
    mask_valid = mask[sim_idx][int(nsims*train_data):int(nsims*train_data)+int(nsims*valid_data)]

    np.savez(f'val_{name}', features=data_norm_valid, mask=mask_valid)
    np.savetxt(f"val_{name}_cond.csv", cond_norm_valid, delimiter=',', header=header, comments='')

    #data_norm_test = data_norm[sim_idx][int(nsims*train_data)+int(nsims*valid_data):]
    #cond_norm_test = cond_norm[sim_idx][int(nsims*train_data)+int(nsims*valid_data):]
    #mask_test = mask[sim_idx][int(nsims*train_data)+int(nsims*valid_data):]

    #np.savez(f'test_{name}', features=data_norm_test, mask=mask_test)
    #np.savetxt(f"test_{name}_cond.csv", cond_norm_test, delimiter=',', header=header, comments='')


    return

if __name__=='__main__':
    main()
