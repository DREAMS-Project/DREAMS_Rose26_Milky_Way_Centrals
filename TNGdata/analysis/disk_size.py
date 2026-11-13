import numpy as np
import os
import DREAMS
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
from readData.DataLoader import DataLoader as DL
from mpi4py import MPI

def lin(x, m, b):
    return m*x+b

def get_disk_size(coords, vels, masses):

    r, d = DREAMS.calc_surface_dens_profile2(coords, vels, masses, 1, 10, npoints=40)

    rcut = r > 3

    max_d = np.max(d[rcut])
    min_d = max_d / np.e**4
    if min_d < .01:
        min_d = .01

    rcut = (r>3) & (d > min_d)

    if np.sum(rcut) < 5:
        return 0

    x = r[rcut]
    y = np.log(d[rcut])
    
    guess = [-0.1, 2]
    popt, pcov = curve_fit(lin, x, y) #, guess)
    m, b = popt
   
    err = np.sqrt(np.sum(np.square(y - lin(x,m, b)))) / len(y)
    scale_len = -1/m

    if err > 0.075:
        rcut = (r > 6) & (d > min_d)
        x = r[rcut]
        y = np.log(d[rcut])
        
        guess = [-0.1, 2]
        popt, pcov = curve_fit(lin, x, y) #, guess)
        m, b = popt
       
        err = np.sqrt(np.sum(np.square(y - lin(x,m, b)))) / len(y)
        scale_len = -1/m

    return scale_len

h = .6909
def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    snap = 99

    path = "../TNG100/"
    tpath = "../TNG100/sublink/"
    name = 'tng100'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloVel', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupVel']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']
    tree_keys = ['SubhaloID','SubhaloGrNr','SnapNum','FirstSubhaloInFOFGroupID','MainLeafProgenitorID','DescendantID','GroupPos']

    #tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

    cat = DL(path, snap, -1, keys=['GroupMass','GroupFirstSub'])
    masses = cat['GroupMass']*1e10/cat.h 
    mcut = (masses > 5e11) & (masses < 3e12)
    first_subs = cat['GroupFirstSub'][mcut].astype(int)

    my_subs = np.array([sub for i,sub in enumerate(first_subs) if i%size == rank]) 

    print(len(my_subs))

    subs = []
    disk_mass = []
    disk_rad = []
    for sub_idx in my_subs[::-1]:
        break

        print(len(disk_mass))

        pcat = DL(path, snap, 4, keys=keys+part_keys, sub_idx=sub_idx)

        coords = (pcat['PartType4/Coordinates'] - pcat['SubhaloPos'])/h
        vels = pcat['PartType4/Velocities'] - pcat['SubhaloVel']
        masses = pcat['PartType4/Masses'] * 1e10/h

        wind_cut = pcat['PartType4/GFM_StellarFormationTime'] > 0
        halo_cut = DREAMS.select_halo_stars(pcat, tree_cat, sub_idx)

        print(len(masses), np.sum(wind_cut), np.sum(halo_cut))

        coords = coords[wind_cut][~halo_cut]
        vels = vels[wind_cut][~halo_cut]
        masses = masses[wind_cut][~halo_cut]

        if len(masses) < 1000:
            print("??", sub_idx)
            continue

        coords, vels = DREAMS.get_rotate_data(coords, vels, masses, face_on=True)

        r = np.sqrt(np.sum(np.square(coords),axis=1))
        epsilon = DREAMS.get_circularity(coords, vels, masses)
        disk_cut = DREAMS.select_disk_stars(epsilon)

        if np.sum(disk_cut & (r>3)) < 5000:
            print("!!!!!", sub_idx, np.sum(disk_cut & (r>3)))
            continue

        scale_len = get_disk_size(coords[disk_cut], vels[disk_cut], masses[disk_cut])

        subs.append(sub_idx)
        disk_mass.append(np.log10(np.sum(masses[disk_cut])))
        disk_rad.append(scale_len)

        
    #np.save(f'np_arrays/subs_{name}_{rank}', subs)
    #np.save(f'np_arrays/disk_size_{name}_{rank}', disk_rad)
    #np.save(f'np_arrays/disk_smass_{name}_{rank}', disk_mass)

    comm.Barrier()
    if rank != 0:
        return

    all_sub = []
    all_rad = []
    all_mass = []
    for rank in range(size):
        subs = np.load(f'np_arrays/subs_{name}_{rank}.npy')
        disk_rad = np.load(f'np_arrays/disk_size_{name}_{rank}.npy')
        disk_mass = np.load(f'np_arrays/disk_smass_{name}_{rank}.npy')
        all_sub.extend(list(subs))
        all_rad.extend(list(disk_rad))
        all_mass.extend(list(disk_mass))

    idx = np.argsort(all_sub)
    np.save(f"np_arrays/subs_{name}", np.array(all_sub)[idx])
    np.save(f"np_arrays/disk_size_{name}", np.array(all_rad)[idx])
    np.save(f"np_arrays/disk_smass_{name}", np.array(all_mass)[idx])

    return


if __name__=='__main__':
    main()
