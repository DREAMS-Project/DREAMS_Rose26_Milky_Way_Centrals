import numpy as np
import os
import DREAMS
from mpi4py import MPI

from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.spatial import KDTree
from scipy.stats import norm
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

h = .6909

def get_vcirc_mass(pcat, cat):

    G = 4.3e-6 #kpc/Msun * (km/s)^2
    all_r = np.linspace(0.3,25,500)

    part_r = []
    part_mass = []
    for pt in [0,1,4,5]:
        r = np.sqrt(np.sum(np.square(pcat[f'PartType{pt}/Coordinates']-cat['GroupPos']),axis=1)) / .6909
        part_mass += list(pcat[f'PartType{pt}/Masses']*1e10/.6909)
        part_r += list(r)

    part_r = np.array(part_r)
    part_mass = np.array(part_mass)

    v_circ = []
    for r in all_r:
        rcut = part_r < r 
        v_circ.append(np.sqrt(G*np.sum(part_mass[rcut]) / r))
            
    return all_r, v_circ

def get_vcirc_pot(pcat, cat):

    part_r = []
    part_pot = []
    for pt in [0,1,4,5]:
        r = np.sqrt(np.sum(np.square(pcat[f'PartType{pt}/Coordinates']-cat['GroupPos']),axis=1)) / h
        part_pot += list(pcat[f'PartType{pt}/Potential'])
        part_r += list(r)

    rcut = np.array(part_r) < 30

    part_r = np.array(part_r)[rcut]
    part_pot = np.array(part_pot)[rcut]

    idx = np.argsort(part_r)

    x = part_r[idx]
    y = part_pot[idx]

    Y2 = savgol_filter(y, 1000, 3)

    interp_x = np.linspace(0,30,1000)
    interp_y = np.interp(interp_x, x, Y2)

    y1 = gaussian_filter(interp_y, 10, 0)

    spl = CubicSpline(interp_x, y1)

    dpdr = spl(interp_x, 1)
    #dpdr = savgol_filter(dpdr, 100, 3)
    v_circ =  np.sqrt(interp_x*dpdr) 

    return interp_x, v_circ

def get_vcirc_pot2(pcat, cat):

    part_r = []
    part_coords = []
    part_pot = []
    for pt in [0,1,4,5]:
        r = np.sqrt(np.sum(np.square(pcat[f'PartType{pt}/Coordinates']-cat['GroupPos']),axis=1)) / h
        part_pot += list(pcat[f'PartType{pt}/Potential'])
        part_r += list(r)
        part_coords += list((pcat[f'PartType{pt}/Coordinates']-cat['GroupPos']) / h)

    rcut = np.array(part_r) < 30

    part_r = np.array(part_r)[rcut]
    part_pot = np.array(part_pot)[rcut]
    part_coords = np.array(part_coords)[rcut]

    tree = KDTree(part_coords)
    gauss = norm(0,1.0)

    all_r = np.linspace(0,30,200)
    all_p = []
    for r in all_r:
        points = DREAMS.fibonacci_sphere(256, r)
        distance, idx = tree.query(points, 64)
        all_p.append(np.average(part_pot[idx], weights=gauss.pdf(distance)))

    r = np.array(all_r)
    p = np.array(all_p)
   
    spl = CubicSpline(r, p)
    dpdr = spl(r, 1)
    dpdr = gaussian_filter(dpdr, 2, 0)
    v_circ =  np.sqrt(r*dpdr) 
    
    return r, v_circ

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
    keys += ['GroupPos']
    part_keys = ['Coordinates', 'Masses', 'Potential']

    all_v = []
    weights = []

    boxes = range(1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])  

    for box in my_boxes:

        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'

        if rank == 0:
            print(box)
        
        #if box in DREAMS.no_disk:
        #    continue 
        
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[0,1,4,5])
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)

        #good_data = np.load("violin_data/weights/disk_smass.npy").astype(int)
        #if box not in good_data:
        #    continue

        #r_mass, v_mass = get_vcirc_mass(galaxy_pcat, galaxy_cat)
        #ax.plot([0]+list(r_mass), [0]+list(v_mass))

        #r_pot, v_pot = get_vcirc_pot(galaxy_pcat, galaxy_cat)
        #ax.plot(r_pot, v_pot, 'k--',alpha=0.6)


        r_pot2, v_pot2 = get_vcirc_pot2(galaxy_pcat, galaxy_cat)

        idx = np.argmin(np.abs(r_pot2-8))

        all_v.append(v_pot2[idx])

        #weight = DREAMS.get_weight(param_path, weight_path, box)
        weights.append(box)

    all_v = np.array(all_v)
    all_y1data = np.array(weights,dtype=float)

    np.save(f"np_arrays/vcirc_{rank}", all_v)
    np.save(f"np_arrays/vcirc_weights_{rank}", all_y1data)

    comm.Barrier()

    if rank > 0:
        return

    vcirc = []
    all_boxes = []
    for rank in range(size):
        vcirc.extend(list(np.load(f"np_arrays/vcirc_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/vcirc_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    vcirc = np.array(vcirc)[idx]
    all_boxes = np.array(all_boxes)[idx]

    print(all_boxes.shape)

    if True:
        np.save('violin_data/disk_vcirc', vcirc)
        np.save('violin_data/weights/disk_vcirc', all_boxes)



    return


if __name__=='__main__':
    main()
