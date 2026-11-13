import numpy as np
import os
import DREAMS
from mpi4py import MPI
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from copy import copy
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def ellipsoidal_flattening(positions, weights=None):
    """
    Fit an ellipsoid to the stellar particle distribution and compute flattening q = c/a.
    
    Parameters:
    - positions: (N, 3) array of star particle positions (x, y, z) centered on galaxy.
    - weights: Optional (N,) array of weights (e.g., stellar mass); defaults to uniform.
    
    Returns:
    - q: flattening (minor-to-major axis ratio c/a)
    - eigenvectors: (3, 3) array of principal axis unit vectors (rows)
    """
    if weights is None:
        weights = np.ones(len(positions))

    # Compute the weighted inertia tensor (second moment)
    I = np.zeros((3, 3))
    for i in range(len(positions)):
        r = positions[i].reshape(3, 1)
        I += weights[i] * r @ r.T

    # Diagonalize
    eigvals, eigvecs = np.linalg.eigh(I)

    # Sort by descending eigenvalue: a > b > c
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Axis lengths proportional to sqrt of eigenvalues
    axis_lengths = np.sqrt(eigvals)
    a, b, c = axis_lengths

    flattening = c / a

    return flattening


h = .6909
def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    snap = 90

    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
    param_path = DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupPos', 'GroupVel']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']

    boxes = range(1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])  

    all_xdata = []
    all_y1data = []
    all_y2data = []
    all_y3data = []
    all_y4data = []
    weights = []
    for box in my_boxes:

        if box == 698:
            continue

        #if box in DREAMS.no_disk:
        #    continue 

        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

        if rank == 0:
            print(box)
        
        if not (os.path.exists(gpath) and os.path.exists(spath) and os.path.exists(tpath)):
            print('Missing Path to Box', box)
            continue
        
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[4])
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
        tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

        if len(tree_cat['SubhaloGrNr']) != len(tree_cat['SnapNum']):
            continue

        ftime = galaxy_pcat['PartType4/GFM_StellarFormationTime']
        wind_cut = ftime > 0

        halo_cut = DREAMS.select_halo_stars(galaxy_pcat, tree_cat, mw_idx)

        coords = (galaxy_pcat['PartType4/Coordinates'] - galaxy_cat['GroupPos'])/h
        vels = galaxy_pcat['PartType4/Velocities'] - galaxy_cat['GroupVel']
        masses = galaxy_pcat['PartType4/Masses'] * 1e10/h

        disk_coords, disk_vels = DREAMS.get_rotate_data(coords[wind_cut][~halo_cut], vels[wind_cut][~halo_cut], masses[wind_cut][~halo_cut], face_on=True)
        r = np.sqrt(np.sum(np.square(disk_coords),axis=1))
        epsilon = DREAMS.get_circularity(disk_coords, disk_vels, masses[wind_cut][~halo_cut])
        disk_cut = DREAMS.select_disk_stars(epsilon, seed=box)

        #has_disk = np.sum(epsilon > 0.7) / len(epsilon) > 0.2
        #if np.sum(disk_cut & (r>3)) < 100 or not has_disk or box in DREAMS.no_disk:
        #    continue

        halo_coords = np.array(list(coords[wind_cut][halo_cut]) + list(coords[wind_cut][~halo_cut][~disk_cut]))
        halo_smass = np.array(list(masses[wind_cut][halo_cut]) + list(masses[wind_cut][~halo_cut][~disk_cut]))

        #r = np.sqrt(np.sum(np.square(coords),axis=1))
        #if np.sum(r > 10) < 100:
        #    print(box,"test")
        #    continue

        phi, theta = DREAMS.get_rotate_data(coords[wind_cut][~halo_cut], vels[wind_cut][~halo_cut], masses[wind_cut][~halo_cut], face_on=True, get_pt=True)

        coords = coords[wind_cut][halo_cut]
        vels = vels[wind_cut][halo_cut]
        masses = masses[wind_cut][halo_cut]

        coords, vels = DREAMS.get_rotate_data(coords, vels, masses, phi=phi, theta=theta) 

        r, d = DREAMS.calc_surface_dens_profile2(coords, vels, masses, 5, 100, npoints=100, rot=False)

        min_d = np.max(d) / 1e5
        rcut = (r>5) & (d > min_d)

        x = np.log10(r[rcut])
        y = np.log10(d[rcut])

        def f(x, a, b):
            return a*x+b

        error = []
        lims = []
        best_err = 100
        best_in = None
        best_out = None
        best_lim = None
        found_at = 0
        for lim in np.linspace(np.min(x)*1.1,np.max(x)*0.9,50):
            rcut = x < lim
            in_popt, pcov = curve_fit(f, x[rcut], y[rcut])
            out_popt, pcov = curve_fit(f, x[~rcut], y[~rcut])

            err_in = np.square(y[rcut] - f(x[rcut], *in_popt))
            err_out = np.square(y[~rcut] - f(x[~rcut], *out_popt))

            this_err = np.sum(err_in) + np.sum(err_out)
            this_err /= np.sum(rcut)
            error.append(this_err)
            lims.append(lim)

            if best_err > this_err:
                best_in = in_popt
                best_out = out_popt
                best_err = this_err
                best_lim = lim

        #print(box, best_in, best_out, best_err)

        #fig,ax = DREAMS.set_plot_params()
        #ax.plot(np.arange(len(error)), error, 'k-')
        #ax.set_yscale("log")
        #fig.savefig(f"plots/halo/break_{box}.pdf", bbox_inches='tight')
        #continue

        # Compute shape
        r = np.sqrt(np.sum(np.square(coords),axis=1))
        r = np.log10(r+1e-10)
        if np.abs(best_in[0] - best_out[0]) < 1:
            best_lim = 1
            best_out = best_in
            inner_cut = (r > np.log10(5))
            q_in = ellipsoidal_flattening(coords[inner_cut], masses[inner_cut])
            q_out = q_in
        else:
            inner_cut = (r < best_lim) & (r > np.log10(5))
            outer_cut = (r > best_lim) & (r < np.log10(100))
            q_in = ellipsoidal_flattening(coords[inner_cut], masses[inner_cut])
            q_out = ellipsoidal_flattening(coords[outer_cut], masses[outer_cut])
            

        if False:

            np.save(f"plots/np_arrays/halo_shape/best_in_{box}", best_in)
            np.save(f"plots/np_arrays/halo_shape/best_out_{box}", best_out)
            np.save(f"plots/np_arrays/halo_shape/x_{box}", x)
            np.save(f"plots/np_arrays/halo_shape/y_{box}", y)
            np.save(f"plots/np_arrays/halo_shape/best_lim_{box}", best_lim)

            #fig, ax = DREAMS.set_plot_params()
            #bl = best_lim
            #print(best_err)
            #ax.plot(x[x<bl], f(x[x<bl],*best_in), 'k-')
            #ax.plot(x[x>bl], f(x[x>bl],*best_out), 'k-')
            #ax.plot(x, y, '.')
            #ax.set_xlim((0.5,2))
            #ax.set_ylim((-8,2))
            #fig.savefig(f"plots/halo/break_{box}.pdf")
            continue

        all_xdata.append(np.power(10,best_lim))
        all_y1data.append(best_in[0])
        all_y2data.append(best_out[0])
        all_y3data.append(q_in)
        all_y4data.append(q_out)

        weights.append(box)

    np.save(f"np_arrays/best_lim_{rank}", np.array(all_xdata))
    np.save(f"np_arrays/best_in_{rank}", np.array(all_y1data))
    np.save(f"np_arrays/best_out_{rank}", np.array(all_y2data))
    np.save(f"np_arrays/q_in_{rank}", np.array(all_y3data))
    np.save(f"np_arrays/q_out_{rank}", np.array(all_y4data))
    np.save(f"np_arrays/halo_shape_weights_{rank}", np.array(weights))

    comm.Barrier()

    if rank > 0:
        return

    all_lim = []
    all_in = []
    all_out = []
    all_qin = []
    all_qout = []
    all_boxes = []
    for rank in range(size):
        all_lim.extend(list(np.load(f"np_arrays/best_lim_{rank}.npy")))
        all_in.extend(list(np.load(f"np_arrays/best_in_{rank}.npy")))
        all_out.extend(list(np.load(f"np_arrays/best_out_{rank}.npy")))
        all_qin.extend(list(np.load(f"np_arrays/q_in_{rank}.npy")))
        all_qout.extend(list(np.load(f"np_arrays/q_out_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/halo_shape_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    all_lim = np.array(all_lim)[idx]
    all_in = np.array(all_in)[idx]
    all_out = np.array(all_out)[idx]
    all_qin = np.array(all_qin)[idx]
    all_qout = np.array(all_qout)[idx]
    all_boxes = np.array(all_boxes)[idx] 
    if True:
        np.save('violin_data/halo_break', all_lim)
        np.save('violin_data/halo_slope_in', all_in)
        np.save('violin_data/halo_slope_out', all_out)
        np.save('violin_data/halo_q_in', all_qin)
        np.save('violin_data/halo_q_out', all_qout)

        np.save('violin_data/weights/halo_break', all_boxes)
        np.save('violin_data/weights/halo_slope_in', all_boxes)
        np.save('violin_data/weights/halo_slope_out', all_boxes)
        np.save('violin_data/weights/halo_q_in', all_boxes)
        np.save('violin_data/weights/halo_q_out', all_boxes)

    return


if __name__=='__main__':
    main()
