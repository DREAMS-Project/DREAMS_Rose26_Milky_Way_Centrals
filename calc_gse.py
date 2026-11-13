import numpy as np
import h5py
import DREAMS
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")


def get_merger_tree(tree_cat, mw_idx, sub_idx, start_snap=90):

    nnodes = len(tree_cat['SubhaloID'])

    target_cut = (tree_cat['SubhaloGrNr'] == mw_idx) & (tree_cat['SnapNum'] == start_snap) & (tree_cat['SubfindID'] == sub_idx)
    target = tree_cat['SubhaloID'][target_cut][0]
    target_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == target][0]

    start = tree_cat['MainLeafProgenitorID'][target_idx]
    start_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == start][0]
    start_snap = tree_cat['SnapNum'][start_idx]

    fof_list = {start_snap:start_idx}
    next_ID = tree_cat['DescendantID'][start_idx]
    next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]
    while next_ID != -1: ## While a descendant exists
        this_snap           = tree_cat['SnapNum'][next_idx]           ## Get my current snapshot
        fof_list[this_snap] = next_idx       ## Get the FOF group of the progenitor
        next_ID = tree_cat['DescendantID'][next_idx]

        if next_ID == -1:
            break

        next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]

    return fof_list

def get_halo_ids(galaxy_pcat, galaxy_cat, tree_cat, mw_idx):
    ftime = galaxy_pcat['PartType4/GFM_StellarFormationTime']
    wind_cut = ftime > 0
    halo_cut = DREAMS.select_halo_stars(galaxy_pcat, tree_cat, mw_idx)
    coords = galaxy_pcat['PartType4/Coordinates']
    r = np.sqrt(np.sum(np.square(coords[wind_cut] - galaxy_cat['GroupPos']),axis=1)) / DREAMS.h
    rcut = (r < 40) & (r > 10)
    central_ids = galaxy_pcat['PartType4/ParticleIDs'][wind_cut][halo_cut & rcut]
    return central_ids

def cartesian_to_spherical_coords(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)    # azimuthal angle
    return r, theta, phi
    
def cartesian_velocities_to_spherical(x, y, z, vx, vy, vz):
    r, theta, phi = cartesian_to_spherical_coords(x, y, z)

    # Spherical unit vectors in Cartesian coordinates
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Compute components
    v_r = (vx * sin_theta * cos_phi + 
           vy * sin_theta * sin_phi + 
           vz * cos_theta)

    v_theta = (vx * cos_theta * cos_phi + 
               vy * cos_theta * sin_phi - 
               vz * sin_theta)

    v_phi = (-vx * sin_phi + vy * cos_phi)

    return v_r, v_theta, v_phi

def get_all_major_merger(box,tree_cat, galaxy_pcat, galaxy_cat, mw_idx):

    nnodes = len(tree_cat['SubhaloID'])

    #initialize the search by finding the target halo in the tree
    target_cut = np.in1d(tree_cat['SubhaloGrNr'], mw_idx) & np.in1d(tree_cat['SnapNum'], 90)
    target  = tree_cat['FirstSubhaloInFOFGroupID'][target_cut][0]
    target_idx = tree_cat['SubhaloID'] == target

    start = tree_cat['MainLeafProgenitorID'][target_idx]
    start_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == start][0]
    start_snap = tree_cat['SnapNum'][start_idx]

    next_ID = tree_cat['DescendantID'][start_idx]
    next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]
    
    #first get the mass of the central for every snapshot
    mass_over_time = dict()
    while next_ID != -1:
        this_snap = tree_cat['SnapNum'][next_idx]
        this_mass = tree_cat['SubhaloMassType'][next_idx,4]*1e10/DREAMS.h
        if this_mass > 0:
            mass_over_time[this_snap] = this_mass
        
        next_ID = tree_cat['DescendantID'][next_idx]
        if next_ID == -1:
            break
        next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]

    next_ID = tree_cat['DescendantID'][start_idx]
    next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]

    last_snap = []
    last_mass = []
    last_ratio = []
    last_fof = []
    last_sub = []
    while next_ID != -1: # While a descendant exists
        other_sat_id = tree_cat['NextProgenitorID'][next_idx]
        main_mass = tree_cat['SubhaloMassType'][next_idx,4]*1e10/DREAMS.h
        
        #search all halos that merge during this snapshot
        while other_sat_id > -1 and main_mass > 0: 
            other_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == other_sat_id][0]
            other_start_id = tree_cat['MainLeafProgenitorID'][other_idx]

            #do the current snapshot
            other_mass = tree_cat['SubhaloMassType'][other_idx,4]*1e10/DREAMS.h
            this_snap = tree_cat['SnapNum'][other_idx]
            this_fof = tree_cat['SubhaloGrNr'][other_idx]
            this_sub = tree_cat['SubfindID'][other_idx]
            
            max_mass = other_mass
            max_snap = this_snap
            max_fof = this_fof
            max_sub = this_sub
            if other_start_id != other_sat_id: #continue if the progenitor has any history
                other_start_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == other_start_id][0]
                other_next_ID = tree_cat['DescendantID'][other_start_idx]
                other_next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == other_next_ID][0]
                while other_next_ID != other_sat_id: #get the maximum mass of the progenitor
                    other_mass = tree_cat['SubhaloMassType'][other_next_idx,4]*1e10/DREAMS.h
                    other_snap = tree_cat['SnapNum'][other_next_idx]
                    other_fof = tree_cat['SubhaloGrNr'][other_next_idx]
                    other_sub = tree_cat['SubfindID'][other_next_idx]
                    if other_mass > max_mass and other_snap in mass_over_time.keys(): #if there is a major merger, save it
                        max_mass = other_mass
                        max_snap = other_snap
                        max_fof = other_fof
                        max_sub = other_sub
                    other_next_ID = tree_cat['DescendantID'][other_next_idx]
                    if other_next_ID == -1:
                        break
                    other_next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == other_next_ID][0]

            #save merger if it happens while the MW exists
            if max_snap in mass_over_time:
                ratio = max_mass / mass_over_time[max_snap]
                if ratio > 1:
                    ratio = 1/ratio

                if ratio > 0.02: #limit to consider major merger
                    last_snap.append(max_snap)
                    last_mass.append(max_mass)
                    last_ratio.append(ratio)
                    last_fof.append(max_fof)
                    last_sub.append(max_sub)

            other_sat_id = tree_cat['NextProgenitorID'][other_idx]
        
        next_ID = tree_cat['DescendantID'][next_idx]
        if next_ID == -1:
            break
        next_idx = np.arange(nnodes)[tree_cat['SubhaloID'] == next_ID][0]
    
    #search for radial mergers in all major mergers
    halo_ids = get_halo_ids(galaxy_pcat, galaxy_cat, tree_cat, mw_idx) #get MW z=0 halo particles

    all_f = []
    all_coords = []
    for i in range(len(last_snap)): #loop over all major mergers

        merger_fof = last_fof[i]
        merger_sub = last_sub[i]

        merger_tree = get_merger_tree(tree_cat, merger_fof, merger_sub, last_snap[i])
        merger_snap = DREAMS.save_snaps[np.argmin(np.abs(last_snap[i] - DREAMS.save_snaps[DREAMS.save_snaps <= last_snap[i]]))]

        #if gal not in previous snap, look for next saved snap
        if merger_snap not in merger_tree:
            late_cut = DREAMS.save_snaps > last_snap[i]
            merger_snap = DREAMS.save_snaps[late_cut][0]

            #only look at mergers if the progenitor exists in a saved snapshot
            if merger_snap not in merger_tree:
                all_f.append(0)
                all_coords.append([[0],[0],[0]])
                print(f"Failed to Find Merger with Mass {last_mass[i]:.2e} in Snapshot {merger_snap}")
                continue
        
        merger_sub = tree_cat['SubfindID'][merger_tree[merger_snap]]
        
        #load all star particles for projenitor at snapshot before peak mass
        snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
        group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{merger_snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{merger_snap:03}.hdf5'
        mcat = DREAMS.load_group_data(gpath, ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs'])     #read DREAMS data for this snapshot
        mpcat = DREAMS.load_particle_data(spath, ['ParticleIDs'], part_types=[4])
        merger_pcat, merger_cat = DREAMS.get_galaxy_data(mpcat, mcat, -1, merger_sub)
        merger_ids = merger_pcat['PartType4/ParticleIDs']

        old_mass = last_mass[i]*0.5
        new_mass = merger_cat['GroupMassType'][4]*1e10/DREAMS.h
        if new_mass < old_mass/2 and old_mass > 1e7:
            print(f"!!!!! Galaxy Has Lost >1/2 Its Mass {last_snap[i]}:{old_mass:.2e} {merger_snap}:{new_mass:.2e} !!!!!")
    
        #calculate fraction of z=0 halo stars that came from progenitor
        all_ids = galaxy_pcat['PartType4/ParticleIDs']
        halo_cut = np.in1d(all_ids, halo_ids)
        merger_cut = np.in1d(halo_ids, merger_ids)
        halo_mass = np.sum(galaxy_pcat['PartType4/Masses'][halo_cut]*1e10/DREAMS.h)
        merger_mass = np.sum(galaxy_pcat['PartType4/Masses'][halo_cut][merger_cut]*1e10/DREAMS.h)
        
        f = merger_mass / halo_mass

        #save z=0 velocities to calc beta
        if merger_mass > 0:
            coords = (galaxy_pcat['PartType4/Coordinates'][halo_cut][merger_cut]-galaxy_cat['GroupPos'])/DREAMS.h
            vels = galaxy_pcat['PartType4/Velocities'][halo_cut][merger_cut]-galaxy_cat['GroupVel']
            vr, vt, vp = cartesian_velocities_to_spherical(coords[:,0], coords[:,1], coords[:,2], vels[:,0], vels[:,1], vels[:,2])
            all_coords.append([vr, vt, vp])
        else:
            all_coords.append([[0],[0],[0]])

        all_f.append(f)

    all_f = np.array(all_f)
    fcut = all_f > 0.1 #only look at mergers that make up 10% of halo
    idx = np.arange(len(last_snap))[fcut]
    
    if np.sum(all_f > 0.5) > 0: #if one merger makes up >50% of stellar halo, choose it
        print("!!!!! Major Merger !!!!!")
        idx = np.array([np.argmax(all_f)])
    
    if len(idx) == 0: #if no mergers are 10% of halo take largest component
        idx = np.array([np.argmax(all_f)])

    #calc beta for all major stellar debris at z=0 (accounts for RA pairs)
    all_vr = []
    all_vt = []
    all_vp = []
    for i in idx:
        all_vr.extend(list(all_coords[i][0]))
        all_vt.extend(list(all_coords[i][1]))
        all_vp.extend(list(all_coords[i][2]))
        
    sp = np.std(all_vp); st = np.std(all_vt); sr = np.std(all_vr)
    beta = 1 - (sp*sp + st*st)/(2*sr*sr)

    if np.isnan(beta):
        beta = 0

    #save merger that makes up most of z=0 halo
    f = np.sum(all_f[idx])
    merger_idx = np.argmax(all_f[idx])
    merger_time = np.array(last_snap)[idx][merger_idx]
    merger_ratio = np.array(last_ratio)[idx][merger_idx]

    mf = all_f[idx][merger_idx]
    mt = np.array(last_snap)[idx][merger_idx]
    mm = np.array(last_mass)[idx][merger_idx]
    mr = np.array(last_ratio)[idx][merger_idx]
    mb = beta

    print(f"Box: {box}\nTotal f: {f:.2f}\nLargest f: {mf:.2f}\nTime: {mt}\nMerger Mass: {mm:.2e}\nMerger Ratio: {mr:.2f}\nMerger Beta: {mb:.2f}\n")
    
    #save: most recent major merger, time of GSE merger, ratio of GSE merger, number of major mergers,
    #      mass of debris from major mergers in z=0 halo, beta for GSE debris
    return np.max(last_snap), merger_time, merger_ratio, len(idx), f, beta

def main():

    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'
    param_path = DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['SubhaloMass', 'SubhaloPos', 'GroupVel', 'GroupMass', 'Group_M_Crit200']
    tree_keys = ['SubhaloGrNr','FirstSubhaloInFOFGroupID', 'SnapNum','SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID', 'NextProgenitorID', 'SubhaloMassType', 'GroupPos']
    part_keys = ['Coordinates', 'Masses', 'GFM_StellarFormationTime', 'BirthPos', 'ParticleIDs', 'Velocities']

    params = DREAMS.get_params_cdm(param_path)
    params = np.array(params)

    boxes = range(1024)

    mass = []
    smass = []
    sim_params = []
    mm_time = []
    gse_time = []
    gse_ratio = []
    ngse = []
    halo_f = []
    halo_beta = []
    good_boxes = []
    for box in boxes:

        if box in DREAMS.no_disk:
            continue
        
        snap = 90
        gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
        spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
        tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

        if len(mass) % 50 == 0:
           print(len(mass))
        
        cat = DREAMS.load_group_data(gpath, keys)     #read DREAMS data
        mw_idx = DREAMS.get_MW_idx(cat)        #find target MW
        sub_idx = cat['GroupFirstSub'][mw_idx]
        tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)

        if len(tree_cat['SubhaloGrNr']) != len(tree_cat['SnapNum']):
            print(box, "!!!")
            continue

        pcat = DREAMS.load_particle_data(spath, part_keys, part_types=[4])
        galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
        
        recent_snap, merger_snap, merger_ratio, nmerger, f, beta = get_all_major_merger(box,tree_cat, galaxy_pcat, galaxy_cat, mw_idx)

        recent_sf = DREAMS.get_scf(group_path, recent_snap)
        merger_sf = DREAMS.get_scf(group_path, merger_snap)

        mm_time.append(recent_sf)
        gse_time.append(merger_sf)
        gse_ratio.append(merger_ratio)
        ngse.append(nmerger)
        halo_f.append(f)
        halo_beta.append(beta)
        
        this_mass = cat['Group_M_Crit200'][mw_idx]*1e10/DREAMS.h
        this_smass = cat['SubhaloMassType'][sub_idx,4]*1e10/DREAMS.h
        mass.append(this_mass)
        smass.append(this_smass)
        
        sim_params.append(params[box])
        good_boxes.append(box)

    mass = np.array(mass)
    smass = np.array(smass)
    sim_params = np.array(sim_params)
    mm_time = np.array(mm_time)
    gse_time = np.array(gse_time)
    gse_ratio = 1/np.array(gse_ratio)
    ngse = np.array(ngse)
    halo_f = np.array(halo_f)
    halo_beta = np.array(halo_beta)
    good_boxes = np.array(good_boxes)

    om = sim_params[:,0]
    s8 = sim_params[:,1]
    sn1 = sim_params[:,2]
    sn2 = sim_params[:,3]
    agn1 = sim_params[:,4]

    data_cond = np.array([np.log10(mass),
                          np.log10(smass),
                          om,
                          s8,
                          np.log10(sn1), 
                          np.log10(sn2), 
                          np.log10(agn1),
                          mm_time,
                          gse_time,
                          gse_ratio,
                          ngse,
                          halo_f,
                          halo_beta
                          ]).T

    header = 'total_mass,stellar_mass,omega_m,sigma_8,log_sn1,log_sn2,log_agn1,mm_time,gse_time,gse_ratio,ngse,halo_f,halo_beta'
    np.savetxt("nehod_data/gse_prob_from_sims.csv", data_cond, delimiter=',', header=header, comments='')
    np.save("nehod_data/gse_prob_boxes_from_sims", good_boxes)

    return

if __name__=="__main__":
    main()
