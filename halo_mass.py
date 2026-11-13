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

        if box == 698:
            continue

        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

        if rank == 0:
            print(box)

        #if box in DREAMS.no_disk:
        #    continue 
        
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[4])
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
        tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

        #if len(tree_cat['SubhaloGrNr']) != len(tree_cat['SnapNum']):
        #    continue

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

        #has_disk = np.sum(epsilon > 0.7) / len(epsilon) > 0.2
        #if np.sum(disk_cut & (r>3)) < 100 or not has_disk or box in DREAMS.no_disk:
        #    continue

        #circ_cut = (epsilon > 0.3) | (r < 3) | np.isnan(epsilon) | (r > 40) 
        #central_smass = np.sum(masses[~circ_cut])

        #fig, ax = DREAMS.set_plot_params()
        #x1 = np.sqrt(np.sum(np.square(coords[wind_cut]),axis=1))
        #y1 = np.cumsum(np.arange(np.sum(wind_cut)))
        #ax.plot(sorted(x1), y1/np.max(y1))
        #x2 = np.sqrt(np.sum(np.square(coords[wind_cut][halo_cut]),axis=1))
        #y2 = np.cumsum(np.arange(np.sum(halo_cut)))
        #ax.plot(sorted(x2), y2/np.max(y2))
        #x3 = np.sqrt(np.sum(np.square(coords[circ_cut]),axis=1))
        #y3 = np.cumsum(np.arange(np.sum(circ_cut)))
        #ax.plot(sorted(x3), y3/np.max(y3))
        #fig.savefig("plots/halo_mass.pdf", bbox_inches='tight')


        if False:

            fig, ax = DREAMS.set_plot_params(nrows=2, ncols=4)
            xpoints = np.linspace(-50,50,128)
            ypoints = np.linspace(-50,50,128)
            zpoints = np.linspace(-50,50,128)

            mesh_points = np.meshgrid(xpoints,ypoints,zpoints)
            mesh_points = np.array(mesh_points)

            x = mesh_points[0].ravel('C')
            y = mesh_points[1].ravel('C')
            z = mesh_points[2].ravel('C')
            sample_points = np.array([x,y,z]).T

            density = DREAMS.calc_density(coords, masses, sample_points, 32)
            image = density.reshape(128, 128, 128, order='C')
            im = ax[0,0].imshow(np.average(image,axis=2), norm=LogNorm(vmin=1e2,vmax=4e6))
            im = ax[1,0].imshow(np.average(image,axis=1), norm=LogNorm(vmin=1e2,vmax=4e6))
            #fig.colorbar(im, ax=ax[0,0])

            density = DREAMS.calc_density(coords[~circ_cut], masses[~circ_cut], sample_points, 32)
            image = density.reshape(128, 128, 128, order='C')
            im = ax[0,1].imshow(np.average(image,axis=2), norm=LogNorm(vmin=1e2,vmax=4e6))
            im = ax[1,1].imshow(np.average(image,axis=1), norm=LogNorm(vmin=1e2,vmax=4e6))
            #fig.colorbar(im, ax=ax[0,1])

            density = DREAMS.calc_density(coords[wind_cut][halo_cut], masses[wind_cut][halo_cut], sample_points, 32)
            image = density.reshape(128, 128, 128, order='C')
            im = ax[0,2].imshow(np.average(image,axis=2), norm=LogNorm(vmin=1e2,vmax=4e6))
            im = ax[1,2].imshow(np.average(image,axis=1), norm=LogNorm(vmin=1e2,vmax=4e6))
            #fig.colorbar(im, ax=ax[0,2])

            density = DREAMS.calc_density(coords[wind_cut][~halo_cut], masses[wind_cut][~halo_cut], sample_points, 32)
            image = density.reshape(128, 128, 128, order='C')
            im = ax[0,3].imshow(np.average(image,axis=2), norm=LogNorm(vmin=1e2,vmax=4e6))
            im = ax[1,3].imshow(np.average(image,axis=1), norm=LogNorm(vmin=1e2,vmax=4e6))

            fig.savefig("plots/halo_finding_test.pdf", bbox_inches='tight')

        #print(f'{central_smass:.2e}')
        #print(f'{np.sum(masses):.2e}')
        #print(f'{np.sum(masses[wind_cut][halo_cut]):.2e}')
        #print(f'{np.sum(masses[~wind_cut]):.2e}')

        halo_coords = np.array(list(coords[wind_cut][halo_cut]) + list(coords[wind_cut][~halo_cut][~disk_cut]))
        halo_smass = np.array(list(masses[wind_cut][halo_cut]) + list(masses[wind_cut][~halo_cut][~disk_cut]))


        #inr = np.sqrt(np.sum(np.square(coords[wind_cut][~halo_cut][~disk_cut]),axis=1))
        #rcut = (inr > 10) & (inr < 45)
        #m = np.log10(np.sum(masses[wind_cut][~halo_cut][~disk_cut][rcut]))
        #frac = np.power(10,m) / np.sum(halo_smass)
        #print(f"{m:.3f} {frac:.3f}")

        r = np.sqrt(np.sum(np.square(halo_coords),axis=1))
        rcut = (r > 10) & (r < 45)

        central_smass = np.sum(halo_smass[rcut])

        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, -1)
        satellite_smass = np.sum(galaxy_cat['SubhaloMassType'][1:,4]) * 1e10/h
        total_smass = galaxy_cat['GroupMassType'][4]*1e10/h - np.sum(masses[~wind_cut])

        #total_smass = frac

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
