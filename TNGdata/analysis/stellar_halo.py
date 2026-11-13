import os
import sys
import glob
import h5py
import numpy as np
from multiprocessing import Pool
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from readData.DataLoader import DataLoader as DL
import DREAMS

# Stellar halo properties for disk-selected TNG centrals, following the DREAMS scripts:
#   halo_central_smass  (halo_mass.py)  log mass of accreted + non-disk in-situ stars at 10 < r < 45 kpc
#   halo_break, halo_slope_in/out, halo_q_in/out  (halo_shape.py)  broken power-law fit to the
#                       accreted-star surface density (5-100 kpc) and c/a inside/outside the break
# Accreted stars follow DREAMS.select_halo_stars: born > 50 kpc from the main-progenitor FoF center.
# Differences from the DREAMS/TNG copies: box h instead of 0.6909, periodic wrapping, correct
# z for snapshot 75, and the main branch read as contiguous SubLink rows.
# usage: python stellar_halo.py tng50|tng100 [ntest]

snap = 99
paths = {'tng50': '../TNG50/', 'tng100': '../TNG100/'}
part_keys = ['Coordinates', 'Velocities', 'Masses', 'GFM_StellarFormationTime', 'BirthPos']

#TNG snapshot redshifts (from DREAMS.track_halo_center, with snap 75 corrected from 0.5)
snap_z = np.array([20.05,14.99,11.98,10.98,10,9.39,9,8.45,8.01,7.6,7.24,7.01,6.49,6.01,5.85,5.53,5.23,5,4.66,4.43,4.18,4.01,3.71,3.49,3.28,3.01,2.9,2.73,2.58,2.44,2.32,2.21,2.10,2.0,1.9,1.82,1.74,1.67,1.6,1.53,1.5,1.41,1.36,1.3,1.25,1.21,1.15,1.11,1.07,1.04,1,0.95,0.92,0.89,0.85,0.82,0.79,0.76,0.73,0.7,0.68,0.64,0.62,0.6,0.58,0.55,0.52,0.5,0.48,0.46,0.44,0.42,0.4,0.38,0.36,0.35,0.33,0.31,0.3,0.27,0.26,0.24,0.23,0.21,0.2,0.18,0.17,0.15,0.14,0.13,0.11,0.1,0.08,0.07,0.06,0.05,0.03,0.02,0.01,0.0])
snap_a = 1/(1+snap_z)

def lin(x, a, b):
    return a*x+b

def wrap(dx, box):
    return (dx + box/2) % box - box/2

def ellipsoidal_flattening(positions, weights):
    #same as halo_shape.ellipsoidal_flattening, vectorized
    I = (positions * weights[:,None]).T @ positions
    eigvals = np.sort(np.linalg.eigvalsh(I))[::-1]
    a, b, c = np.sqrt(eigvals)
    return c / a

def build_tree_index(path, subs, fname):

    if os.path.isfile(fname):
        return dict(np.load(fname))

    files, rows, lens, found = [], [], [], []
    tfiles = sorted(glob.glob(f'{path}/sublink/tree_extended.*.hdf5'), key=lambda f: int(f.split('.')[-2]))
    for tf in tfiles:
        with h5py.File(tf, 'r') as f:
            snapnum = f['SnapNum'][:]
            subfind = f['SubfindID'][:]
            idx = np.where((snapnum == snap) & np.isin(subfind, subs))[0]
            if len(idx) == 0:
                continue
            sid = f['SubhaloID'][:][idx]
            mlp = f['MainLeafProgenitorID'][:][idx]
        files.extend([int(tf.split('.')[-2])]*len(idx))
        rows.extend(idx)
        lens.extend(mlp - sid + 1)
        found.extend(subfind[idx])
        print(tf, len(idx), flush=True)

    found = np.array(found)
    order = np.argsort(found)
    index = {'sub_id': found[order], 'tree_file': np.array(files)[order], 'row': np.array(rows)[order], 'nrows': np.array(lens)[order]}
    np.savez(fname, **index)
    return index

def select_halo_stars(path, tree_file, row, nrows, ftime, bpos, h, box):

    with h5py.File(f'{path}/sublink/tree_extended.{tree_file}.hdf5', 'r') as f:
        snaps = f['SnapNum'][row:row+nrows]
        gpos = f['GroupPos'][row:row+nrows]

    #main branch is ordered from z=0 backwards; unwrap positions relative to z=0
    gpos = gpos[0] + wrap(gpos - gpos[0], box)
    a = snap_a[snaps]
    order = np.argsort(a)
    a, gpos = a[order], gpos[order]

    halo_pos = np.array([CubicSpline(a, gpos[:,k])(ftime) for k in range(3)]).T
    halo_dist = np.sqrt(np.sum(np.square(wrap(halo_pos - bpos, box)), axis=1)) / h

    return halo_dist > 50

def fit_break(coords, vels, masses):

    r, d = DREAMS.calc_surface_dens_profile2(coords, vels, masses, 5, 100, npoints=100, rot=False)

    min_d = np.max(d) / 1e5
    rcut = (r>5) & (d > min_d)
    x = np.log10(r[rcut])
    y = np.log10(d[rcut])

    best_err = 100
    best_in, best_out, best_lim = None, None, None
    for lim in np.linspace(np.min(x)*1.1, np.max(x)*0.9, 50):
        rcut = x < lim
        in_popt, pcov = curve_fit(lin, x[rcut], y[rcut])
        out_popt, pcov = curve_fit(lin, x[~rcut], y[~rcut])

        this_err = np.sum(np.square(y[rcut] - lin(x[rcut], *in_popt))) + np.sum(np.square(y[~rcut] - lin(x[~rcut], *out_popt)))
        this_err /= np.sum(rcut)

        if best_err > this_err:
            best_in, best_out, best_err, best_lim = in_popt, out_popt, this_err, lim

    return best_lim, best_in, best_out

def process(args):

    path, h, box, sub_idx, gpos, gvel, tree_file, row, nrows = args
    np.random.seed(sub_idx)   #select_disk_stars draws random numbers
    out = np.full(6, np.nan)

    try:
        pcat = DL(path, snap, 4, keys=part_keys, sub_idx=int(sub_idx))

        ftime = pcat['PartType4/GFM_StellarFormationTime']
        wind_cut = ftime > 0
        bpos = pcat['PartType4/BirthPos'][wind_cut]
        halo_cut = select_halo_stars(path, tree_file, row, nrows, ftime[wind_cut], bpos, h, box)

        coords = wrap(pcat['PartType4/Coordinates'] - gpos, box)[wind_cut]/h
        vels = (pcat['PartType4/Velocities'] - gvel)[wind_cut]
        masses = pcat['PartType4/Masses'][wind_cut] * 1e10/h

        #in-situ disk stars (halo_mass.py / halo_shape.py)
        disk_coords, disk_vels = DREAMS.get_rotate_data(coords[~halo_cut], vels[~halo_cut], masses[~halo_cut], face_on=True)
        epsilon = DREAMS.get_circularity(disk_coords, disk_vels, masses[~halo_cut])
        disk_cut = DREAMS.select_disk_stars(epsilon)

        halo_coords = np.concatenate((coords[halo_cut], coords[~halo_cut][~disk_cut]))
        halo_smass = np.concatenate((masses[halo_cut], masses[~halo_cut][~disk_cut]))
        r = np.sqrt(np.sum(np.square(halo_coords), axis=1))
        rcut = (r > 10) & (r < 45)
        out[0] = np.log10(np.sum(halo_smass[rcut]))

        #accreted-star profile and shape, in the frame of the in-situ disk (halo_shape.py)
        phi, theta = DREAMS.get_rotate_data(coords[~halo_cut], vels[~halo_cut], masses[~halo_cut], face_on=True, get_pt=True)
        acc_coords, acc_vels = DREAMS.get_rotate_data(coords[halo_cut], vels[halo_cut], masses[halo_cut], phi=phi, theta=theta)
        acc_masses = masses[halo_cut]

        best_lim, best_in, best_out = fit_break(acc_coords, acc_vels, acc_masses)

        r = np.log10(np.sqrt(np.sum(np.square(acc_coords), axis=1)) + 1e-10)
        if np.abs(best_in[0] - best_out[0]) < 1:
            best_lim = 1
            best_out = best_in
            inner_cut = r > np.log10(5)
            q_in = ellipsoidal_flattening(acc_coords[inner_cut], acc_masses[inner_cut])
            q_out = q_in
        else:
            inner_cut = (r < best_lim) & (r > np.log10(5))
            outer_cut = (r > best_lim) & (r < np.log10(100))
            q_in = ellipsoidal_flattening(acc_coords[inner_cut], acc_masses[inner_cut])
            q_out = ellipsoidal_flattening(acc_coords[outer_cut], acc_masses[outer_cut])

        out[1:] = [np.power(10, best_lim), best_in[0], best_out[0], q_in, q_out]

    except Exception as e:
        print(f'failed {sub_idx}: {type(e).__name__}: {e}', flush=True)

    return out

def main():

    name = sys.argv[1]
    ntest = int(sys.argv[2]) if len(sys.argv) > 2 else None
    path = paths[name]

    with h5py.File(sorted(glob.glob(f'{path}/groups_{snap:03}/*.hdf5'))[0], 'r') as f:
        h = float(f['Header'].attrs['HubbleParam'])
        box = float(f['Header'].attrs['BoxSize'])

    cat = np.load(f'np_arrays/catalog_{name}.npz')
    disk = cat['has_disk']
    subs, grnr = cat['sub_id'][disk], cat['grnr'][disk]

    gcat = DL(path, snap, -1, keys=['GroupPos', 'GroupVel'])
    tree = build_tree_index(path, subs, f'np_arrays/tree_index_{name}.npz')
    assert np.array_equal(tree['sub_id'], subs), 'centrals missing from SubLink'

    args = [(path, h, box, subs[i], gcat['GroupPos'][grnr[i]], gcat['GroupVel'][grnr[i]],
             tree['tree_file'][i], tree['row'][i], tree['nrows'][i]) for i in range(len(subs))]
    if ntest is not None:
        args = args[:ntest]
    print(name, 'h =', h, 'box =', box, 'N =', len(args), flush=True)

    with Pool(14) as pool:
        res = np.array(pool.map(process, args, chunksize=1))

    names = ['halo_central_smass', 'halo_break', 'halo_slope_in', 'halo_slope_out', 'halo_q_in', 'halo_q_out']
    for i, n in enumerate(names):
        print(n, np.round(np.nanpercentile(res[:,i], [0,16,50,84,100]), 3))

    suffix = '_test' if ntest is not None else ''
    np.savez(f'np_arrays/stellar_halo_{name}{suffix}', sub_id=subs[:len(args)], **{n: res[:,i] for i, n in enumerate(names)})

    return

if __name__=='__main__':
    main()
