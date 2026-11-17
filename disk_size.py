import numpy as np
import os
import DREAMS
from mpi4py import MPI
from scipy.optimize import curve_fit
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def lin(x, m, b):
    return m*x+b

def get_circularity(coords, vels, masses):

    Lz = coords[:,0]*masses*vels[:,1] - coords[:,1]*masses*vels[:,0]

    Lmax = np.sqrt(np.sum(np.square(vels), axis=1)) * masses * np.sqrt(np.sum(np.square(coords), axis=1))

    Lcut = Lmax != 0
    epsilon = Lz[Lcut] / Lmax[Lcut]

    out = np.ones(Lcut.shape)
    out[Lcut] = epsilon

    return out

def get_disk_size(box, coords, vels, masses):

    #r, d = DREAMS.calc_surface_dens_profile(coords, vels, masses, 1, 20, part_types=[4], r_step=1.02)
    r, d = DREAMS.calc_surface_dens_profile2(coords, vels, masses, 1, 10, npoints=40)

    rcut = r > 3

    max_d = np.max(d[rcut])
    min_d = max_d / np.e**4
    if min_d < .01:
        min_d = .01

    rcut = (r>3) & (d > min_d)

    if np.sum(rcut) < 5:
        return 0

    if False:
        x = r
        y = np.log(d)
        fig, ax = DREAMS.set_plot_params()
        ax.plot(x, y, '.')
        ax.set_xlim((0,30))
        ax.set_ylim((-5,4))
        fig.savefig(f"plots/disk_size/box_{box}.pdf")
        stop
        return 0

    x = r[rcut]
    y = np.log(d[rcut])
    
    guess = [-0.1, 2]
    popt, pcov = curve_fit(lin, x, y) #, guess)
    m, b = popt
   
    err = np.sqrt(np.sum(np.square(y - lin(x,m, b)))) / len(y)
    scale_len = -1/m

    if False: #scale_len < 0.2:
        fig, ax = DREAMS.set_plot_params()
        print(box, scale_len, err)
        ax.plot(x, lin(x,m,b), 'k-')
        ax.plot(x, y, '.')
        ax.set_xlim((0,30))
        ax.set_ylim((-3,4))
        fig.savefig(f"plots/disk_size/box_{box}.pdf")
        stop

    if err > 0.075:
        rcut = (r > 6) & (d > min_d)
        x = r[rcut]
        y = np.log(d[rcut])
        
        guess = [-0.1, 2]
        popt, pcov = curve_fit(lin, x, y) #, guess)
        m, b = popt
       
        err = np.sqrt(np.sum(np.square(y - lin(x,m, b)))) / len(y)
        scale_len = -1/m

        #print(box, err)

    if True:
        np.save(f"plots/np_arrays/disk_size/x_{box}", x)
        np.save(f"plots/np_arrays/disk_size/y_{box}", y)
        np.save(f"plots/np_arrays/disk_size/mb_{box}", np.array([m,b]))

    return scale_len

def select_disk_stars(epsilon):

    bins = np.linspace(-1,1,40)
    bin_centers = (bins[1:]+bins[:-1])/2
    tot_cnt, be, bn = binned_statistic(epsilon, epsilon, 'count', bins)

    scut = epsilon < 0
    sph_cnt, be, bn = binned_statistic(epsilon[scut], epsilon[scut], 'count', bins[:20])
    sph_cnt = np.array(list(tot_cnt[:20]) + list(sph_cnt[::-1]))

    phalo = sph_cnt / tot_cnt

    rnd = np.random.rand(len(epsilon))
    is_disk = np.zeros((len(epsilon)),dtype=bool)
    for i in range(len(epsilon)):
        idx = np.arange(len(bins))[epsilon[i] > bins][-1] 
        if rnd[i] > phalo[idx]:
           is_disk[i] = True 

    return is_disk

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
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']

    boxes = range(1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])  

    disk_mass = []
    disk_rad = []
    disk_weight = []
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
            disk_weight.append(box)
            disk_mass.append(6)
            disk_rad.append(0.3)
            continue

        scale_len = get_disk_size(box,coords[disk_cut], vels[disk_cut], masses[disk_cut])

        disk_weight.append(box)
        disk_mass.append(np.log10(np.sum(masses[disk_cut])))
        disk_rad.append(scale_len)

        
    all_xdata = np.array(disk_rad,dtype=float)
    all_ydata = np.array(disk_mass,dtype=float)
    all_y1data = np.array(disk_weight,dtype=float)

    np.save(f"np_arrays/disk_rad_{rank}", all_xdata)
    np.save(f"np_arrays/disk_mass_{rank}", all_ydata)
    np.save(f"np_arrays/bulge_mass_weights_{rank}", all_y1data)

    comm.Barrier()

    if rank > 0:
        return

    disk_rad = []
    disk_mass = []
    all_boxes = []
    for rank in range(size):
        disk_rad.extend(list(np.load(f"np_arrays/disk_rad_{rank}.npy")))
        disk_mass.extend(list(np.load(f"np_arrays/disk_mass_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/bulge_mass_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    disk_rad = np.array(disk_rad)[idx]
    disk_mass = np.array(disk_mass)[idx]
    all_boxes = np.array(all_boxes)[idx] 

    if True:
        np.save('violin_data/disk_size_old', disk_rad)
        np.save('violin_data/disk_smass_old', disk_mass)
        np.save('violin_data/weights/disk_size_old', all_boxes)
        np.save('violin_data/weights/disk_smass_old', all_boxes)


    return


if __name__=='__main__':
    main()
