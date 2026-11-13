import numpy as np
import os
import DREAMS
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

h = .6909
def main():

    snap = 90

    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
    param_path = DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
    weight_path = 'violin_data/binned_weights.npy'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupPos', 'GroupVel', 'SubhaloPos','SubhaloVel','SubhaloMass']

    boxes = range(1024)

    data = np.zeros((803, 50, 8))
    mask = np.zeros((803, 50), dtype=bool)

    cond = np.loadtxt("data/gse_prob_cond.csv", delimiter=',', skiprows=1)
    cond = np.round(cond, 6)
    params = np.loadtxt(DREAMS_DATA + "/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt", skiprows=1)

    all_nsat = []
    for box in boxes:

        print(box)
        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW

        sat_group_cut = cat['SubhaloGrNr'] == mw_idx
        sat_smass_cut = cat['SubhaloMassType'][:,4]*1e10 / h > 1e7
        sat_cut = sat_group_cut & sat_smass_cut
        central_cut = np.arange(len(sat_cut))[sat_cut][0]
        sat_cut[central_cut] = False

        nsats = np.sum(sat_cut)

        if nsats == 0:
            nsats = 1
            coords = np.array([np.random.rand()]*3)
            vels = np.array([np.random.rand()]*3)
            masses = 8
            smasses = 6
        else:
            coords = cat['SubhaloPos'][sat_cut] - cat['GroupPos'][mw_idx]
            vels = cat['SubhaloVel'][sat_cut] - cat['GroupVel'][mw_idx]
            masses = cat['SubhaloMass'][sat_cut] * 1e10 / h
            smasses = cat['SubhaloMassType'][sat_cut,4] * 1e10 / h

            smasses[smasses==0] = 1e6
            
        om = params[box, 0]
        if om not in cond[:,3]:
            continue

        idx = cond[:,3] == om
        idx = np.arange(len(cond))[idx][0]
        print(box, idx)

        data[idx, :nsats, :3] = coords / 1000
        data[idx, :nsats, 3:6] = vels / 100
        data[idx, :nsats, 6] = np.log10(masses)
        data[idx, :nsats, 7] = np.log10(smasses)

        mask[idx, :nsats] = [True]*nsats

        all_nsat.append(nsats)

    np.savez('data/gse_prob', features=data, mask=mask)
    np.save("data/nsat", np.log10(np.array(all_nsat).astype(int)))
   
    return


if __name__=='__main__':
    main()
