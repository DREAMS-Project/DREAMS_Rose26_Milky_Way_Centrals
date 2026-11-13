import numpy as np
import os
from matplotlib.colors import LogNorm
from readData.DataLoader import DataLoader as DL

h = .6909
def main():

    snap = 99

    path = "../TNG100/"
    #path = "/orange/paul.torrey/IllustrisTNG/Runs/L75n1820TNG/"

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupVel', 'SubhaloSFR', 'SubhaloMass','GroupMass']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos', 'StarFormationRate']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']

    cat = DL(path, snap, -1, keys=keys)     #read DREAMS data

    masses = cat['GroupMass']*1e10/cat.h 
    mcut = (masses > 5e11) & (masses < 3e12)
  
    first_subs = cat['GroupFirstSub'][mcut].astype(int)
    nsubs = cat['GroupNsubs'][mcut].astype(int)

    sfr = cat['SubhaloSFR'][first_subs]
    bmass = (cat['SubhaloMassType'][:,0] + cat['SubhaloMassType'][:,4])[first_subs]*1e10/h
    mass = cat['SubhaloMass'][first_subs]*1e10/h
    smass = cat['SubhaloMassType'][:,4][first_subs]*1e10/h

    sat_smass = []
    for i in range(len(first_subs)):
        ssat = np.sum(cat['SubhaloMassType'][first_subs[i]+1:first_subs[i]+nsubs[i],4])
        sat_smass.append(ssat*1e10/h)
    sat_smass = np.array(sat_smass)
    sat_smass[sat_smass==0] = 5e5

    sfr[sfr==0] = np.power(10,-4.)

    name = 'tng50'

    np.save(f"np_arrays/sfr_{name}", np.log10(sfr))
    np.save(f"np_arrays/bmass_{name}", np.log10(bmass))
    np.save(f"np_arrays/bmass_frac_{name}", np.log10(bmass/mass))
    np.save(f"np_arrays/smass_{name}", np.log10(smass))
    np.save(f"np_arrays/sat_smass_{name}", np.log10(sat_smass))

    return


if __name__=='__main__':
    main()
