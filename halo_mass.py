import numpy as np
import os
import DREAMS
from mpi4py import MPI
from matplotlib.colors import LogNorm
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def get_circularity(coords, vels, masses):

    Lz = coords[:,0]*masses*vels[:,1] - coords[:,1]*masses*vels[:,0]

    Lmax = np.sqrt(np.sum(np.square(vels), axis=1)) * masses * np.sqrt(np.sum(np.square(coords), axis=1))

    Lcut = Lmax != 0
    epsilon = Lz[Lcut] / Lmax[Lcut]

    out = np.ones(Lcut.shape)
    out[Lcut] = epsilon

    return out

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
    weights = []
    for box in my_boxes:

        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

        if rank == 0:
            print(box)

        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[4])
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
        tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

        ftime = galaxy_pcat['PartType4/GFM_StellarFormationTime']
        wind_cut = ftime > 0

        halo_cut = DREAMS.select_halo_stars(galaxy_pcat, tree_cat, mw_idx)

        coords = (galaxy_pcat['PartType4/Coordinates'] - galaxy_cat['GroupPos'])/h
        vels = galaxy_pcat['PartType4/Velocities'] - galaxy_cat['GroupVel']
        masses = galaxy_pcat['PartType4/Masses'] * 1e10/h

        disk_coords, disk_vels = DREAMS.get_rotate_data(coords[wind_cut][~halo_cut], vels[wind_cut][~halo_cut], masses[wind_cut][~halo_cut], face_on=True)
        r = np.sqrt(np.sum(np.square(disk_coords),axis=1))
        epsilon = get_circularity(disk_coords, disk_vels, masses[wind_cut][~halo_cut])
        disk_cut = DREAMS.select_disk_stars(epsilon, seed=box)

        if np.sum(disk_cut & (r>3)) < 5000:
            continue


        halo_coords = np.array(list(coords[wind_cut][halo_cut]) + list(coords[wind_cut][~halo_cut][~disk_cut]))
        halo_smass = np.array(list(masses[wind_cut][halo_cut]) + list(masses[wind_cut][~halo_cut][~disk_cut]))

        r = np.sqrt(np.sum(np.square(halo_coords),axis=1))
        rcut = (r > 10) & (r < 45)

        central_smass = np.sum(halo_smass[rcut])

        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, -1)
        satellite_smass = np.sum(galaxy_cat['SubhaloMassType'][1:,4]) * 1e10/h
        total_smass = galaxy_cat['GroupMassType'][4]*1e10/h - np.sum(masses[~wind_cut])

        all_xdata.append(total_smass)
        all_y1data.append(central_smass)
        all_y2data.append(satellite_smass)

        weights.append(box)

    np.save(f"np_arrays/total_smass_{rank}", np.array(all_xdata))
    np.save(f"np_arrays/central_smass_{rank}", np.array(all_y1data))
    np.save(f"np_arrays/satellite_smass_{rank}", np.array(all_y2data))
    np.save(f"np_arrays/halo_mass_weights_{rank}", np.array(weights))

    comm.Barrier()

    if rank > 0:
        return

    all_tot_smass = []
    all_cen_smass = []
    all_sat_smass = []
    all_boxes = []
    for rank in range(size):
        all_tot_smass.extend(list(np.load(f"np_arrays/total_smass_{rank}.npy")))
        all_cen_smass.extend(list(np.load(f"np_arrays/central_smass_{rank}.npy")))
        all_sat_smass.extend(list(np.load(f"np_arrays/satellite_smass_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/halo_mass_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    all_tot_smass = np.array(all_tot_smass)[idx]
    all_cen_smass = np.array(all_cen_smass)[idx]
    all_sat_smass = np.array(all_sat_smass)[idx]
    all_boxes = np.array(all_boxes)[idx] 

    if True:
        np.save('violin_data/halo_tot_smass', np.log10(all_tot_smass))
        np.save('violin_data/halo_central_smass', np.log10(all_cen_smass))
        np.save('violin_data/halo_sat_smass', np.log10(all_sat_smass))

        np.save('violin_data/weights/halo_tot_smass', all_boxes)
        np.save('violin_data/weights/halo_central_smass', all_boxes)
        np.save('violin_data/weights/halo_sat_smass', all_boxes)


    return


if __name__=='__main__':
    main()
