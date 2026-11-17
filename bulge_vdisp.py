import numpy as np
import os
import DREAMS
from mpi4py import MPI
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")


h = .6909
def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    snap = 90
    part_types = [4]

    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
    param_path = DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupPos','GroupVel']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos', 'Potential']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']

    all_data = []

    boxes = range(1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])  

    no_disk = [0,27,60,67,79,112,113,152,175,233,264,282,288,343,347,356,364,381,391,399,454,503,507,542,553,578,605,624,630,637,650,659,683,745,746,753,757,768,813,862,866,874,890,895,955,964,979,1008]

    vel_dispx = []
    vel_dispy = []
    vel_dispz = []
    vel_disp_tot = []
    all_reff = []
    weights = []
    for box in my_boxes:


        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'
        weight_path = 'violin_data/binned_weights.npy'

        if rank == 0:
            print(box)
        
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=part_types)
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
        tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

        coords = (galaxy_pcat['PartType4/Coordinates'] - galaxy_cat['GroupPos'])/h
        vels = galaxy_pcat['PartType4/Velocities'] - galaxy_cat['GroupVel']
        masses = galaxy_pcat['PartType4/Masses'] * 1e10/h

        wind_cut = galaxy_pcat['PartType4/GFM_StellarFormationTime'] > 0
        halo_cut = DREAMS.select_halo_stars(galaxy_pcat, tree_cat, mw_idx)

        coords = coords[wind_cut][~halo_cut]
        vels = vels[wind_cut][~halo_cut]
        masses = masses[wind_cut][~halo_cut]

        coords, vels = DREAMS.get_rotate_data(coords, vels, masses, face_on=True)

        r = np.sqrt(np.sum(np.square(coords),axis=1))
        epsilon = DREAMS.get_circularity(coords, vels, masses)
        disk_cut = DREAMS.select_disk_stars(epsilon, seed=box)

        if np.sum(disk_cut & (r>3)) < 5000:
            continue

        coords = (galaxy_pcat['PartType4/Coordinates'] - galaxy_cat['GroupPos'])/h
        vels = galaxy_pcat['PartType4/Velocities'] - galaxy_cat['GroupVel']
        masses = galaxy_pcat['PartType4/Masses'] * 1e10/h
        potential = galaxy_pcat['PartType4/Potential']

        bulge_coords = np.array(list(coords[wind_cut][halo_cut]) + list(coords[wind_cut][~halo_cut][~disk_cut]))
        bulge_vels = np.array(list(vels[wind_cut][halo_cut]) + list(vels[wind_cut][~halo_cut][~disk_cut]))
        bulge_masses = np.array(list(masses[wind_cut][halo_cut]) + list(masses[wind_cut][~halo_cut][~disk_cut]))
        bulge_pots = np.array(list(potential[wind_cut][halo_cut]) + list(potential[wind_cut][~halo_cut][~disk_cut]))


        disk_coords = coords[wind_cut][~halo_cut][disk_cut]
        disk_masses = masses[wind_cut][~halo_cut][disk_cut]
        disk_vels = vels[wind_cut][~halo_cut][disk_cut]
        disk_pots = potential[wind_cut][~halo_cut][disk_cut]

        bulge_cut = DREAMS.select_bulge_stars(bulge_coords, bulge_masses, bulge_vels, bulge_pots, disk_coords, disk_masses, disk_vels, disk_pots)

        coords = bulge_coords[bulge_cut]
        vels = bulge_vels[bulge_cut]
        masses = bulge_masses[bulge_cut]

        reff = DREAMS.get_reff(box, coords, vels, masses)
        r = np.sqrt(np.sum(np.square(coords),axis=1))

        rcut = r < 5 #reff*3 #*3 for VVV region
        rsort = np.argsort(r[rcut])
        csum = np.cumsum(masses[rcut][rsort])
        rhalf_idx = np.argmin(np.abs(csum/np.max(csum) - 0.5))
        rhalf = r[rcut][rsort][rhalf_idx]
        rcut = r < rhalf

        speeds = np.sqrt(np.sum(np.square(vels[rcut]),axis=1)) 

        vel_dispx.append(np.std(vels[rcut,0]))
        vel_dispy.append(np.std(vels[rcut,1]))
        vel_dispz.append(np.std(vels[rcut,2]))
        vel_disp_tot.append(np.sqrt(np.mean(np.square(speeds))))
        all_reff.append(reff)

        #weight = DREAMS.get_weight(param_path, weight_path, box)
        weights.append(box)

    #vel_dispx = np.array(vel_dispx,dtype=float)
    #vel_dispy = np.array(vel_dispy,dtype=float)
    #vel_dispz = np.array(vel_dispz,dtype=float)
    vel_disp_tot = np.array(vel_disp_tot,dtype=float)
    all_y1data = np.array(weights,dtype=float)

    np.save(f"np_arrays/vdisp_{rank}", vel_disp_tot)
    np.save(f"np_arrays/bulge_vdisp_weights_{rank}", np.array(weights))

    comm.Barrier()

    if rank > 0:
        return

    all_vdisp = []
    all_boxes = []
    for rank in range(size):
        all_vdisp.extend(list(np.load(f"np_arrays/vdisp_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/bulge_vdisp_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    all_vdisp = np.array(all_vdisp)[idx]
    all_boxes = np.array(all_boxes)[idx]

    print(all_boxes.shape)

    if True:
        np.save('violin_data/bulge_vdisp', all_vdisp)
        np.save('violin_data/weights/bulge_vdisp', all_boxes)



    return


if __name__=='__main__':
    main()
