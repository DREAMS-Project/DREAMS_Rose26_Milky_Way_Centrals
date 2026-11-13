import numpy as np
import pandas as pd

def main():

    name = 'master_cdm'

    header = 'stellar_mass,total_mass,bh_mass,vel_disp,OH_frac,tf_vel,tf_mass,omega_m,sigma_8,log_sn1,log_sn2,log_agn1'

    data = np.load(f'{name}.npz')
    cond = pd.read_csv(f'{name}_cond.csv')

    data_norm_raw = data['features']
    mask_raw = data['mask']
    cond_norm = cond.to_numpy()

    train_data = 0.8
    valid_data = 0.2
    test_data = 0.0

    nsims = len(cond)
    sim_idx = np.arange(nsims)
    np.random.default_rng(13).shuffle(sim_idx)  # Release: seeded split

    size = list(data_norm_raw.shape)
    size[0] = nsims
    data_norm = np.empty(size)
    data_norm[:len(data_norm_raw)] = data_norm_raw
    data_norm[len(data_norm_raw):] = data_norm_raw[:nsims-len(data_norm_raw)]
    mask = np.empty(size[:2])
    mask[:len(data_norm_raw)] = mask_raw
    mask[len(data_norm_raw):] = mask_raw[:nsims-len(data_norm_raw)]

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
