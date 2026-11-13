import numpy as np
import os
import DREAMS
from mpi4py import MPI
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from copy import copy
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def lin(x, m, b):
    return m*x+b

h = .6909
def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    snap = 90
    part_types = [0,1,2,4,5]

    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
    param_path = DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupPos','GroupVel']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos', 'Potential']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']

    boxes = range(0,1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])  

    save_masses = {pt:[] for pt in part_types}
    save_masses[-1] = []
    all_reff = []
    weights = []
    for box in my_boxes:

        if box == 698:
            continue

        #if not box in DREAMS.no_disk and not box in DREAMS.no_bulge:
        #    continue

        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

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

        coords = (galaxy_pcat['PartType4/Coordinates'] - galaxy_cat['GroupPos'])/h
        vels = galaxy_pcat['PartType4/Velocities'] - galaxy_cat['GroupVel']
        masses = galaxy_pcat['PartType4/Masses'] * 1e10/h
        potential = galaxy_pcat['PartType4/Potential']

        halo_coords = np.array(list(coords[wind_cut][halo_cut]) + list(coords[wind_cut][~halo_cut][~disk_cut]))
        halo_vels = np.array(list(vels[wind_cut][halo_cut]) + list(vels[wind_cut][~halo_cut][~disk_cut]))
        halo_masses = np.array(list(masses[wind_cut][halo_cut]) + list(masses[wind_cut][~halo_cut][~disk_cut]))
        halo_pots = np.array(list(potential[wind_cut][halo_cut]) + list(potential[wind_cut][~halo_cut][~disk_cut]))

        disk_coords = coords[wind_cut][~halo_cut][disk_cut]
        disk_masses = masses[wind_cut][~halo_cut][disk_cut]
        disk_vels = vels[wind_cut][~halo_cut][disk_cut]
        disk_pots = potential[wind_cut][~halo_cut][disk_cut]

        bulge_cut = DREAMS.select_bulge_stars(halo_coords, halo_masses, halo_vels, halo_pots, disk_coords, disk_masses, disk_vels, disk_pots)
        coords = halo_coords[bulge_cut]
        vels = halo_vels[bulge_cut]
        masses = halo_masses[bulge_cut]

        max_reff, mean_reff = DREAMS.get_reff2(box, coords, vels, masses)

        r = np.sqrt(np.sum(np.square(coords),axis=1))
        bulge_mass = np.sum(masses[r<5])
        
        tot_mass = 0
        for pt in part_types:
            r = np.sqrt(np.sum(np.square(galaxy_pcat[f'PartType{pt}/Coordinates']-galaxy_cat['GroupPos']),axis=1))/h
            rcut = r < mean_reff*3
            pt_mass = np.sum(galaxy_pcat[f'PartType{pt}/Masses'][rcut]*1e10/h)
            tot_mass += pt_mass
            if pt == 4:
                save_masses[4].append(bulge_mass)
            if pt == 5:
                if pt_mass < 1e4:
                    pt_mass = np.random.default_rng(box).random()*3+2  # placeholder for no BH (see ISSUES.md)
                save_masses[pt].append(pt_mass)
            else:
                save_masses[pt].append(pt_mass)
        save_masses[-1].append(tot_mass)
        
        all_reff.append(max_reff)
        weights.append(box)

    all_reff = np.array(all_reff,dtype=float)

    dyn_mass = np.array(save_masses[-1],dtype=float)
    stellar_mass = np.array(save_masses[4],dtype=float)
    bh_mass = np.array(save_masses[5],dtype=float)
    
    dm1_mass = np.array(save_masses[1],dtype=float)
    dm2_mass = np.array(save_masses[2],dtype=float)
    dm_frac = (dm1_mass + dm2_mass) / dyn_mass

    np.save(f"np_arrays/all_reff_{rank}", all_reff)
    np.save(f"np_arrays/dyn_mass_{rank}", dyn_mass)
    np.save(f"np_arrays/stellar_mass_{rank}", stellar_mass)
    np.save(f"np_arrays/bh_mass_{rank}", bh_mass)
    np.save(f"np_arrays/dm_frac_{rank}", dm_frac)
    np.save(f"np_arrays/bulge_mass_weights_{rank}", np.array(weights))

    comm.Barrier()

    if rank > 0:
        return

    all_reff = []
    all_dyn = []
    all_smass = []
    all_bh = []
    all_dm = []
    all_boxes = []
    for rank in range(size):
        all_reff.extend(list(np.load(f"np_arrays/all_reff_{rank}.npy")))
        all_dyn.extend(list(np.load(f"np_arrays/dyn_mass_{rank}.npy")))
        all_smass.extend(list(np.load(f"np_arrays/stellar_mass_{rank}.npy")))
        all_bh.extend(list(np.load(f"np_arrays/bh_mass_{rank}.npy")))
        all_dm.extend(list(np.load(f"np_arrays/dm_frac_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/bulge_mass_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    all_reff = np.array(all_reff)[idx]
    all_dyn = np.array(all_dyn)[idx]
    all_smass = np.array(all_smass)[idx]
    all_bh = np.array(all_bh)[idx]
    all_dm = np.array(all_dm)[idx]
    all_boxes = np.array(all_boxes)[idx] 

    print(all_boxes.shape)
    print(all_boxes)

    fig, ax = DREAMS.set_plot_params()
    bins = np.linspace(0.2, 1.3, 30)
    ax.hist(all_reff, bins=bins)
    ax.set_xlabel("R$_\mathrm{bulge}$")
    ax.set_ylabel("Sim Count")
    fig.savefig("plots/bulge_reff_exsitu.pdf", bbox_inches='tight')

    fig, ax = DREAMS.set_plot_params()
    bins = np.linspace(8, 11, 30)
    ax.hist(np.log10(all_smass), bins=bins)
    ax.set_xlabel("M$_\mathrm{bulge}$")
    ax.set_ylabel("Sim Count")
    fig.savefig("plots/bulge_smass_exsitu.pdf", bbox_inches='tight')

    if True:
        np.save('violin_data/bulge_reff_old', all_reff)
        np.save('violin_data/bulge_mass_old', np.log10(all_dyn))
        np.save('violin_data/bulge_smass_old', np.log10(all_smass))
        np.save('violin_data/bulge_bhmass_old', np.log10(all_bh))
        np.save('violin_data/bulge_dm_frac_old', all_dm)

        np.save('violin_data/weights/bulge_reff_old', all_boxes)
        np.save('violin_data/weights/bulge_mass_old', all_boxes)
        np.save('violin_data/weights/bulge_smass_old', all_boxes)
        np.save('violin_data/weights/bulge_bhmass_old', all_boxes)
        np.save('violin_data/weights/bulge_dm_frac_old', all_boxes)

    return


if __name__=='__main__':
    main()
