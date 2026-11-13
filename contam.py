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
    keys += ['Group_R_Crit200','Group_M_Crit200']
    part_keys = ['Coordinates','Masses']
    tree_keys = []

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
        weight_path = 'violin_data/binned_weights.npy'

        if rank == 0:
            print(box) 

        #if box in DREAMS.no_disk:
        #    continue 
 
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[2])
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, -1)

        pos = galaxy_cat['GroupPos']
        r = np.sqrt(np.sum(np.square(galaxy_pcat['PartType2/Coordinates']-pos),axis=1))
        rcut = r < galaxy_cat['Group_R_Crit200']

        masses = galaxy_pcat['PartType2/Masses']
        contam = np.sum(masses[rcut]) / galaxy_cat['Group_M_Crit200']

        print(box, contam)
        
        all_xdata.append(contam)
        weights.append(box)       

    all_xdata = np.array(all_xdata)
    all_y3data = np.array(weights)

    np.save(f"np_arrays/contam_{rank}", all_xdata)
    np.save(f"np_arrays/contam_weights_{rank}", all_y3data)

    comm.Barrier()

    if rank > 0:
        return

    bmass = []
    bmass_frac = []
    sfr = []
    all_boxes = []
    for rank in range(size):
        bmass.extend(list(np.load(f"np_arrays/contam_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/contam_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    bmass = np.array(bmass)[idx]
    all_boxes = np.array(all_boxes)[idx]

    if True:
        np.save('violin_data/halo_contam', bmass)
        np.save('violin_data/weights/halo_contam', all_boxes)

    return


if __name__=='__main__':
    main()
