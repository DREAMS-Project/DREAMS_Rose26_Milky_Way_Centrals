import numpy as np
import os
import DREAMS
from mpi4py import MPI
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
import matplotlib.pyplot as plt
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def calc_density(coords, masses, x, y, z):

    xshape = len(x)
    yshape = len(y)
    zshape = len(z)

    if len(masses) == 0:
        return None

    mesh_points = np.meshgrid(x,y,z)
    mesh_points = np.array(mesh_points)

    x = mesh_points[0].ravel('C')
    y = mesh_points[1].ravel('C')
    z = mesh_points[2].ravel('C')
    sample_points = np.array([x,y,z]).T

    density = DREAMS.calc_density(coords, masses, sample_points, 32)
    image = density.reshape(yshape, xshape, zshape, order='C')

    return image


h = .6909
def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    snap = 90

    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
    param_path = DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'
    weight_path = 'violin_data/binned_weights.npy'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupPos', 'GroupVel']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos', 'Potential']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']

    boxes = range(1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])  

    part_types = [0,1,2,4,5]

    npix = 128

    disk_mass = []
    disk_rad = []
    disk_weight = []

    total = 0
    no_disk = []
    no_bulge = []
    no_halo = []
    for box in my_boxes:

        if box == 698:
            continue

        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

        if rank == 0:
            print(box)
        
        if not (os.path.exists(gpath) and os.path.exists(spath)):
            continue
        
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=part_types)
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
        tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

        if len(tree_cat['SubhaloGrNr']) != len(tree_cat['SnapNum']):
            continue

        E_particles, Ecut, eta, star_masses = DREAMS.calc_Energy_and_eta(galaxy_pcat, galaxy_cat, part_types, box)

        #single_component
        if Ecut is None:
            halo_cut = DREAMS.select_halo_stars(galaxy_pcat, tree_cat, mw_idx)
            if np.median(eta) < 0.5:
                print(box, np.median(eta), "Pure Spheroid")
                
                halo_indices = halo_cut
                bulge_indices = ~halo_cut
                
                pseudo_indices = np.zeros(eta.shape[0],dtype=bool)
                thin_disk_indices = np.zeros(eta.shape[0],dtype=bool)
                thick_disk_indices = np.zeros(eta.shape[0],dtype=bool)
            else:
                print(box, np.median(eta), "Pure Disk")

                halo_indices = np.zeros(eta.shape[0],dtype=bool)
                bulge_indices = np.zeros(eta.shape[0],dtype=bool)
                
                pseudo_indices = np.zeros(eta.shape[0],dtype=bool)
                thin_disk_indices = eta > 0.7
                thick_disk_indices = eta < 0.7

        else:

            bulge_indices, pseudo_indices, thin_disk_indices, thick_disk_indices, halo_indices = DREAMS.get_component_indices(E_particles, Ecut, eta, star_masses)

        has_disk = True
        has_bulge = True
        has_halo = True
        if len(thin_disk_indices)+len(thick_disk_indices) < 1000 or np.sum(thin_disk_indices) + np.sum(thick_disk_indices)== 0:
            has_disk = False
        if len(bulge_indices)+len(pseudo_indices) < 1000 or np.sum(bulge_indices) + np.sum(pseudo_indices) == 0:
            has_bulge = False
        if len(halo_indices) < 1000 or np.sum(halo_indices) == 0:
            has_halo = False

        total += 1
        if not has_disk:
            no_disk.append(box)
        if not has_bulge:
            no_bulge.append(box)
        if not has_halo:
            no_halo.append(box)

    
    print(rank, 'Disk', len(no_disk), 'Bulge', len(no_bulge), 'Halo', len(no_halo))
    print(rank, 'No Disk', no_disk)
    print(rank, 'No Bulge', no_bulge)
    print(rank, 'No Halo', no_halo)
    return       

    if False:

        wind_cut = galaxy_pcat['PartType4/GFM_StellarFormationTime'] > 0
        
        coords = (galaxy_pcat['PartType4/Coordinates'][wind_cut] - galaxy_cat['GroupPos'])/h
        vels = galaxy_pcat['PartType4/Velocities'][wind_cut] - galaxy_cat['GroupVel']
        masses = galaxy_pcat['PartType4/Masses'][wind_cut] * 1e10/h
    
        if len(thin_disk_indices) > 2000 and np.sum(thin_disk_indices>0):
            phi, theta = DREAMS.get_rotate_data(coords[thin_disk_indices], vels[thin_disk_indices], masses[thin_disk_indices], face_on=True, get_pt=True)
            coords, vels = DREAMS.get_rotate_data(coords, vels, masses, phi=phi, theta=theta)
            #print(box, 'Rotating Thin', len(thin_disk_indices), len(thick_disk_indices))
        elif len(thick_disk_indices) > 2000 and np.sum(thick_disk_indices>0):
            phi, theta = DREAMS.get_rotate_data(coords[thick_disk_indices], vels[thick_disk_indices], masses[thick_disk_indices], face_on=True, get_pt=True)
            coords, vels = DREAMS.get_rotate_data(coords, vels, masses, phi=phi, theta=theta)
            #print(box, 'Rotating Thick', len(thin_disk_indices), len(thick_disk_indices))
        else:
            #print(box, 'Not Rotating', len(thin_disk_indices), len(thick_disk_indices))
            pass

        xpoints = np.linspace(-20,20,npix)
        ypoints = np.array(list(np.linspace(-20,20,npix)))
        zpoints = np.linspace(-20,20,npix//2)
        thin_face_image = calc_density(coords[thin_disk_indices], masses[thin_disk_indices], xpoints, ypoints, zpoints)
        thick_face_image = calc_density(coords[thick_disk_indices], masses[thick_disk_indices], xpoints, ypoints, zpoints)

        xpoints = np.linspace(-5,5,npix)
        ypoints = np.array(list(np.linspace(-5,5,npix)))
        zpoints = np.linspace(-5,5,npix//2)
        bulge_image = calc_density(coords[bulge_indices], masses[bulge_indices], xpoints, ypoints, zpoints)
        pseudo_image = calc_density(coords[pseudo_indices], masses[pseudo_indices], xpoints, ypoints, zpoints)

        
        xpoints = np.linspace(-50,50,npix)
        ypoints = np.array(list(np.linspace(-50,50,npix)))
        zpoints = np.linspace(-50,50,npix//2)
        halo_image = calc_density(coords[halo_indices], masses[halo_indices], xpoints, ypoints, zpoints) 

        fig, ax = DREAMS.set_plot_params(1,5,figsize=(25,5))
        for i,im in enumerate([bulge_image,pseudo_image,thin_face_image,thick_face_image,halo_image]):
            if im is not None:
                ax[i].imshow(np.average(im, axis=2), norm=LogNorm())
        #ax[1].imshow(np.average(pseudo_image, axis=2), norm=LogNorm())
        #ax[2].imshow(np.average(thin_face_image, axis=2), norm=LogNorm())
        #ax[3].imshow(np.average(thick_face_image, axis=2), norm=LogNorm())
        #ax[4].imshow(np.average(halo_image, axis=2), norm=LogNorm())
        fig.savefig(f"plots/components/box_{box}.pdf")
        plt.close(fig)

        #continue



        disk_coords = coords[wind_cut][~halo_cut]
        disk_vels = vels[wind_cut][~halo_cut]
        disk_masses = masses[wind_cut][~halo_cut]
        disk_pots = potential[wind_cut][~halo_cut]

        disk_coords, disk_vels = DREAMS.get_rotate_data(disk_coords, disk_vels, disk_masses, face_on=True)
        epsilon = DREAMS.get_circularity(disk_coords, disk_vels, disk_masses)
        disk_cut = DREAMS.select_disk_stars(epsilon, seed=box)

        disk_coords, disk_vels = DREAMS.get_rotate_data(disk_coords[disk_cut], disk_vels[disk_cut], disk_masses[disk_cut], face_on=True)
        xpoints = np.linspace(-20,20,npix)
        ypoints = np.array(list(np.linspace(-20,20,npix)))
        zpoints = np.linspace(-20,20,npix//2)
        face_image = calc_density(disk_coords, disk_masses[disk_cut], xpoints, ypoints, zpoints) 

        disk_coords, disk_vels = DREAMS.get_rotate_data(disk_coords, disk_vels, disk_masses[disk_cut], edge_on=True)
        xpoints = np.linspace(-20,20,npix)
        ypoints = np.array(list(np.linspace(-20,20,npix)))
        zpoints = np.linspace(-20,20,npix//2)
        edge_image = calc_density(disk_coords, disk_masses[disk_cut], xpoints, ypoints, zpoints) 

        halo_coords = np.array(list(coords[wind_cut][halo_cut]) + list(coords[wind_cut][~halo_cut][~disk_cut]))
        halo_vels = np.array(list(vels[wind_cut][halo_cut]) + list(vels[wind_cut][~halo_cut][~disk_cut]))
        halo_masses = np.array(list(masses[wind_cut][halo_cut]) + list(masses[wind_cut][~halo_cut][~disk_cut]))
        halo_pots = np.array(list(potential[wind_cut][halo_cut]) + list(potential[wind_cut][~halo_cut][~disk_cut]))

        bulge_cut = DREAMS.select_bulge_stars(halo_coords, halo_masses, halo_vels, halo_pots, disk_coords, disk_masses[disk_cut], disk_vels, disk_pots[disk_cut])
        #reff = DREAMS.get_reff(box, halo_coords[bulge_cut], halo_vels[bulge_cut], halo_masses[bulge_cut])
        #r = np.sqrt(np.sum(np.square(halo_coords[bulge_cut]),axis=1))
        #rcut = r < reff*3

        xpoints = np.linspace(-5,5,npix)
        ypoints = np.array(list(np.linspace(-5,5,npix)))
        zpoints = np.linspace(-5,5,npix//2)
        bulge_image = calc_density(halo_coords[bulge_cut], halo_masses[bulge_cut], xpoints, ypoints, zpoints) 

        xpoints = np.linspace(-50,50,npix)
        ypoints = np.array(list(np.linspace(-50,50,npix)))
        zpoints = np.linspace(-50,50,npix//2)
        halo_image = calc_density(halo_coords[~bulge_cut], halo_masses[~bulge_cut], xpoints, ypoints, zpoints) 

        fig, ax = DREAMS.set_plot_params(1,4,figsize=(20,5))
        ax[0].imshow(np.average(bulge_image, axis=2), norm=LogNorm())
        ax[1].imshow(np.average(face_image, axis=2), norm=LogNorm())
        ax[2].imshow(np.average(edge_image, axis=2), norm=LogNorm())
        ax[3].imshow(np.average(halo_image, axis=2), norm=LogNorm())
        fig.savefig(f"plots/components/box_{box}.pdf")


    return


if __name__=='__main__':
    main()
