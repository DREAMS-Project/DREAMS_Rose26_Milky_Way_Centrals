import sys
import glob
import h5py
import numpy as np
from multiprocessing import Pool
from readData.DataLoader import DataLoader as DL

# Builds one per-galaxy table for a TNG box, keyed by central subhalo ID, with
# properties defined the same way as the DREAMS scripts (halo_mass.py, gas_mass.py):
#   smass  = GroupMassType[4] - wind mass of the central   (DREAMS halo_tot_smass)
#   bmass  = central SubhaloMassType[0] + [4]              (DREAMS halo_bmass)
#   sfr    = central SubhaloSFR, floored at 1e-4           (DREAMS halo_sfr)
# usage: python build_catalog.py tng50|tng100

snap = 99
mmin, mmax = 5e11, 3e12   #superset of the disk-size selection; tighter cuts applied in compare_data.py
h_dreams = .6909          #h used (incorrectly) when disk_size.py was run on TNG

paths = {'tng50': '../TNG50/', 'tng100': '../TNG100/'}

def get_h(path):
    fname = sorted(glob.glob(f'{path}/groups_{snap:03}/*.hdf5'))[0]
    with h5py.File(fname, 'r') as f:
        return float(f['Header'].attrs['HubbleParam'])

def wind_mass(args):
    path, sub_idx = args
    pcat = DL(path, snap, 4, keys=['Masses', 'GFM_StellarFormationTime'], sub_idx=int(sub_idx))
    wind_cut = pcat['PartType4/GFM_StellarFormationTime'] <= 0
    return np.sum(pcat['PartType4/Masses'][wind_cut])

def main():

    name = sys.argv[1]
    path = paths[name]
    h = get_h(path)

    keys = ['GroupMass', 'GroupMassType', 'Group_M_Crit200', 'GroupFirstSub', 'GroupNsubs']
    keys += ['SubhaloMassType', 'SubhaloMass', 'SubhaloSFR']
    cat = DL(path, snap, -1, keys=keys)

    mfof = cat['GroupMass']*1e10/h
    mcut = (mfof > mmin) & (mfof < mmax) & (cat['GroupNsubs'] > 0)
    grnr = np.arange(len(mfof))[mcut]
    first_subs = cat['GroupFirstSub'][mcut].astype(int)
    print(name, 'h =', h, 'N halos =', len(first_subs))

    with Pool(14) as pool:
        wind = np.array(pool.map(wind_mass, [(path, s) for s in first_subs], chunksize=4))

    smass = (cat['GroupMassType'][mcut,4] - wind)*1e10/h
    smass_sub = cat['SubhaloMassType'][first_subs,4]*1e10/h
    bmass = (cat['SubhaloMassType'][first_subs,0] + cat['SubhaloMassType'][first_subs,4])*1e10/h
    mass = cat['SubhaloMass'][first_subs]*1e10/h
    sfr = cat['SubhaloSFR'][first_subs].copy()
    sfr[sfr==0] = np.power(10,-4.)

    #disk results from disk_size.py, rescaled from h=0.6909 to the box h
    disk_subs = np.load(f'np_arrays/subs_{name}.npy')
    disk_size = np.load(f'np_arrays/disk_size_{name}.npy') * h_dreams/h
    disk_smass = np.load(f'np_arrays/disk_smass_{name}.npy') + np.log10(h_dreams/h)

    has_disk = np.isin(first_subs, disk_subs)
    didx = np.searchsorted(disk_subs, first_subs)
    didx[~has_disk] = 0
    rdisk = np.where(has_disk, disk_size[didx], np.nan)
    mdisk = np.where(has_disk, disk_smass[didx], np.nan)

    missing = np.sum(~np.isin(disk_subs, first_subs))
    print('disk galaxies:', np.sum(has_disk), '| disk subs outside mass cut:', missing)

    np.savez(f'np_arrays/catalog_{name}',
             sub_id=first_subs, grnr=grnr, h=h,
             log_mfof=np.log10(mfof[mcut]),
             log_m200c=np.log10(cat['Group_M_Crit200'][mcut]*1e10/h),
             smass=np.log10(smass), smass_sub=np.log10(smass_sub), wind_mass=wind*1e10/h,
             bmass=np.log10(bmass), bmass_frac=np.log10(bmass/mass), sfr=np.log10(sfr),
             has_disk=has_disk, disk_size=rdisk, disk_smass=mdisk,
             disk_fit_ok=has_disk & (rdisk > 0))

    return

if __name__=='__main__':
    main()
