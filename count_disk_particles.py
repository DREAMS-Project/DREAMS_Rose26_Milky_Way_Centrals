"""Count disk star particles for every DREAMS CDM MW-mass galaxy.

Galaxies without a resolved disk are excluded from the disk-galaxy sample (Sec. 4.1 / App. C:
"these galaxies do not contain at least N star particles that would be selected as part of the
disk outside of 3 kpc"). This script records, per box, the number of disk star particles (all
radii and at r > 3 kpc) for both decompositions used in the paper:

  * stochastic circularity (the paper's method; DREAMS.select_disk_stars, seeded with the box
    number so the selection matches the property scripts)
  * Zana et al. (2022) energy-circularity method (thin + thick disk), as in disk_size_new.py

so that the particle threshold defining DREAMS.no_disk can be chosen and documented.

Run from the repository root:
    mpirun -n <N> python count_disk_particles.py
Output:
    np_arrays/disk_counts.npy, shape (1024, 6), columns
        [box, tree_ok, n_stoch, n_stoch_r3, n_zana, n_zana_r3]
    (-1 marks a count that could not be computed for that box)
    and prints the resulting DREAMS.no_disk list for DREAMS.DISK_MIN_PARTICLES (5000).
    With the released code this flags 189 boxes (835 disk galaxies remain); it also flags all 187
    boxes found by the original 5e3 check (the 2 extra come from the per-box seeding).
"""
import numpy as np
import DREAMS
from mpi4py import MPI
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

h = .6909
R_MIN = 3.0  # kpc


def stochastic_counts(galaxy_pcat, galaxy_cat, tree_cat, mw_idx, box):
    """Disk stars from the stochastic circularity method (same steps as disk_size.py)."""
    coords = (galaxy_pcat['PartType4/Coordinates'] - galaxy_cat['GroupPos'])/h
    vels = galaxy_pcat['PartType4/Velocities'] - galaxy_cat['GroupVel']
    masses = galaxy_pcat['PartType4/Masses'] * 1e10/h

    wind_cut = galaxy_pcat['PartType4/GFM_StellarFormationTime'] > 0
    halo_cut = DREAMS.select_halo_stars(galaxy_pcat, tree_cat, mw_idx)  # True = ex-situ

    coords = coords[wind_cut][~halo_cut]
    vels = vels[wind_cut][~halo_cut]
    masses = masses[wind_cut][~halo_cut]

    coords, vels = DREAMS.get_rotate_data(coords, vels, masses, face_on=True)
    r = np.sqrt(np.sum(np.square(coords), axis=1))
    epsilon = DREAMS.get_circularity(coords, vels, masses)
    disk_cut = DREAMS.select_disk_stars(epsilon, seed=box)

    return int(np.sum(disk_cut)), int(np.sum(disk_cut & (r > R_MIN)))


def zana_counts(galaxy_pcat, galaxy_cat, part_types, box):
    """Thin + thick disk stars from the Zana et al. (2022) method (same steps as disk_size_new.py)."""
    E_particles, Ecut, eta, star_masses = DREAMS.calc_Energy_and_eta(galaxy_pcat, galaxy_cat, part_types, box)

    disk_mask = np.zeros(len(star_masses), dtype=bool)
    if Ecut is None:  # single-component galaxy
        if np.median(eta) >= 0.5:  # "pure disk": every star is thin or thick disk
            disk_mask[:] = True
    else:
        _, _, thin, thick, _ = DREAMS.get_component_indices(E_particles, Ecut, eta, star_masses)
        disk_mask[np.asarray(thin, dtype=int)] = True
        disk_mask[np.asarray(thick, dtype=int)] = True

    wind_cut = galaxy_pcat['PartType4/GFM_StellarFormationTime'] > 0
    coords = (galaxy_pcat['PartType4/Coordinates'][wind_cut] - galaxy_cat['GroupPos'])/h
    r = np.sqrt(np.sum(np.square(coords), axis=1))

    return int(np.sum(disk_mask)), int(np.sum(disk_mask & (r > R_MIN)))


def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size

    snap = 90
    snap_path = DREAMS_DATA + '/Sims/CDM/MW_zooms/SB5/'
    group_path = DREAMS_DATA + '/FOF_Subfind/CDM/MW_zooms/SB5/'

    keys = ['GroupMassType', 'GroupLenType', 'GroupPos', 'SubhaloGrNr', 'SubhaloPos', 'SubhaloMassType', 'SubhaloLenType', 'GroupFirstSub', 'GroupNsubs']
    keys += ['GroupPos', 'GroupVel']
    part_keys = ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime', 'BirthPos', 'Potential']
    tree_keys = ['FirstSubhaloInFOFGroupID', 'FirstProgenitorID', 'SnapNum', 'SubhaloGrNr', 'FirstProgenitorID', 'GroupPos']
    tree_keys += ['SubfindID', 'SubhaloID', 'MainLeafProgenitorID', 'DescendantID']
    part_types = [0,1,2,4,5]

    boxes = range(1024)
    my_boxes = np.array([box for i,box in enumerate(boxes) if i%size == rank])

    rows = []
    for box in my_boxes:
        row = [box, -1, -1, -1, -1, -1]
        try:
            gpath = f'{group_path}/box_{box}/fof_subhalo_tab_{snap:03}.hdf5'
            spath = f'{snap_path}/box_{box}/snap_{snap:03}.hdf5'
            tpath = f'{group_path}/box_{box}/tree_extended.hdf5'

            cat = DREAMS.load_group_data(gpath, keys)
            pcat = DREAMS.load_particle_data(spath, part_keys, part_types=part_types)
            mw_idx = DREAMS.get_MW_idx(cat)
            galaxy_pcat, galaxy_cat = DREAMS.get_galaxy_data(pcat, cat, mw_idx, 0)
            tree_cat = DREAMS.load_sublink_cat(tpath, tree_keys)
            row[1] = int(len(tree_cat['SubhaloGrNr']) == len(tree_cat['SnapNum']))
        except Exception as e:
            print(f"[rank {rank}] box {box}: load failed: {e!r}", flush=True)
            rows.append(row)
            continue

        try:
            row[2], row[3] = stochastic_counts(galaxy_pcat, galaxy_cat, tree_cat, mw_idx, box)
        except Exception as e:
            print(f"[rank {rank}] box {box}: stochastic failed: {e!r}", flush=True)
        try:
            row[4], row[5] = zana_counts(galaxy_pcat, galaxy_cat, part_types, box)
        except Exception as e:
            print(f"[rank {rank}] box {box}: zana failed: {e!r}", flush=True)

        print(f"[rank {rank}] box {box}: {row[1:]}", flush=True)
        rows.append(row)

    np.save(f"np_arrays/disk_counts_{rank}", np.array(rows, dtype=np.int64).reshape(-1, 6))
    comm.Barrier()

    if rank > 0:
        return

    allrows = np.concatenate([np.load(f"np_arrays/disk_counts_{r}.npy") for r in range(size)])
    allrows = allrows[np.argsort(allrows[:, 0])]
    np.save("np_arrays/disk_counts", allrows)
    print("saved np_arrays/disk_counts.npy", allrows.shape)

    # The disk-galaxy sample: copy this list into DREAMS.no_disk
    thr = DREAMS.DISK_MIN_PARTICLES
    no_disk = sorted(int(b) for b in allrows[allrows[:, 3] < thr, 0])
    print(f"no_disk ({len(no_disk)} boxes with < {thr} stochastic disk particles at r > {R_MIN} kpc):")
    print(no_disk)

    return


if __name__ == '__main__':
    main()
