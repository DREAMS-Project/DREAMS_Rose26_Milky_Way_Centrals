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
    keys += ['GroupPos', 'GroupVel', 'SubhaloSFR', 'SubhaloMass']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos', 'StarFormationRate']
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
        weight_path = 'violin_data/binned_weights.npy'

        if rank == 0:
            print(box) 

        #if box in DREAMS.no_disk:
        #    continue 
 
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[3])
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)

        sfr = galaxy_cat['SubhaloSFR']
        bmass = np.sum(galaxy_cat['SubhaloMassType'][[0,4]])*1e10/h
        mass = galaxy_cat['SubhaloMass']*1e10/h

        if sfr == 0:
            sfr = np.power(10,-4.)
        
        all_xdata.append(np.log10(bmass))
        all_y1data.append(np.log10(bmass/mass))
        all_y2data.append(np.log10(sfr))

        #weight = DREAMS.get_weight(param_path, weight_path, box)
        weights.append(box)       

    all_xdata = np.array(all_xdata)
    all_y1data = np.array(all_y1data)
    all_y2data = np.array(all_y2data)
    all_y3data = np.array(weights)

    np.save(f"np_arrays/bmass_{rank}", all_xdata)
    np.save(f"np_arrays/bmass_frac_{rank}", all_y1data)
    np.save(f"np_arrays/sfr_{rank}", all_y2data)
    np.save(f"np_arrays/gas_mass_weights_{rank}", all_y3data)

    comm.Barrier()

    if rank > 0:
        return

    bmass = []
    bmass_frac = []
    sfr = []
    all_boxes = []
    for rank in range(size):
        bmass.extend(list(np.load(f"np_arrays/bmass_{rank}.npy")))
        bmass_frac.extend(list(np.load(f"np_arrays/bmass_frac_{rank}.npy")))
        sfr.extend(list(np.load(f"np_arrays/sfr_{rank}.npy")))
        all_boxes.extend(list(np.load(f"np_arrays/gas_mass_weights_{rank}.npy")))

    idx = np.argsort(all_boxes)

    bmass = np.array(bmass)[idx]
    bmass_frac = np.array(bmass_frac)[idx]
    sfr = np.array(sfr)[idx]
    all_boxes = np.array(all_boxes)[idx]

    if True:
        np.save('violin_data/halo_bmass', bmass)
        np.save('violin_data/halo_bmass_frac', bmass_frac)
        np.save('violin_data/halo_sfr', sfr)

        np.save('violin_data/weights/halo_bmass', all_boxes)
        np.save('violin_data/weights/halo_bmass_frac', all_boxes)
        np.save('violin_data/weights/halo_sfr', all_boxes)

    return


if __name__=='__main__':
    main()
