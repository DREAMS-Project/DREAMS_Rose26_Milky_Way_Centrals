"""Table 2: the values that the Fig 3 violins are normalised to.

For each property this prints, for the disk-galaxy sample (DREAMS.get_disk_sample()):
  * the 1st and 99th percentiles of the simulated values (unweighted, index-based), and
  * the median of the weighted distribution (weights from nehod_weights/arya_weights_binned.npy),
exactly as violin.normalize_data() maps them to -1, 0 and +1 in Fig 3, together with the
observed Milky Way values from Bland-Hawthorn & Gerhard (2016).

It also prints the unweighted min / mean / max. NOTE: the published Table 2 (labelled 1st / 50th /
99th percentiles) actually lists the unweighted min / mean / max of the 803-galaxy sample (the values
violin.py prints), see ISSUES.md.

Run from the repository root after the property scripts and the weights pipeline:
    python make_table2.py            # prints a table and writes table2.csv
"""
import numpy as np
import os

import DREAMS
from violin import calc_weights

# (name, label, units, log?) -- same order as violin.py / Fig 3
PROPS = [
    ('halo_tot_smass', 'M*', 'Msun', True),
    ('halo_sat_smass', 'M*,sat', 'Msun', True),
    ('halo_bmass', 'Mbary', 'Msun', True),
    ('halo_sfr', 'SFR', 'Msun/yr', True),
    ('bulge_reff', 'Reff', 'kpc', False),
    ('bulge_mass', 'MTOT,bulge', 'Msun', True),
    ('bulge_smass', 'M*,bulge', 'Msun', True),
    ('bulge_bhmass', 'MBH', 'Msun', True),
    ('bulge_vdisp', 'sigma*,bulge', 'km/s', False),
    ('disk_vcirc', 'Vcirc', 'km/s', False),
    ('disk_smass', 'M*,disk', 'Msun', True),
    ('disk_size', 'Rdisk', 'kpc', False),
    ('halo_central_smass', 'M*,halo', 'Msun', True),
    ('halo_break', 'r_s', 'kpc', False),
    ('halo_slope_in', 'alpha_in', '', False),
    ('halo_slope_out', 'alpha_out', '', False),
    ('halo_q_in', 'q_in', '', False),
    ('halo_q_out', 'q_out', '', False),
]

# Observed MW values (Bland-Hawthorn & Gerhard 2016), as in violin.py
OBS_MEAN = [10.699, 9.431, 10.9294, 0.2175, 0.70, 10.265, 10.190, 6.623, 113, 238, 10.613, 2.51, 8.740, 25, -2.5, -4.35, 0.65, 0.8]
OBS_HIGH = [10.778, 9.544, 10.9918,  0.265, 0.77, 10.281, 10.230, 6.643, 116, 253, 10.711, 2.97, 8.845, 35, -2.2, -3.70, 0.70, 0.9]
OBS_LOW  = [10.602, 9.279, 10.8670, 0.1644, 0.63, 10.248, 10.146, 6.602, 110, 223, 10.485, 2.06, 8.602, 15, -2.8, -5.00, 0.60, 0.7]

# Fig 3 uses the stochastic-circularity decomposition ('_old' outputs of bulge_mass.py / disk_size.py)
FILE_SUFFIX = {n: '_old' for n in ['bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'disk_smass', 'disk_size']}


def load_property(name, weights, disk_sample):
    """Return (values, per-galaxy weights) with the same cuts as violin.py."""
    fname = name + FILE_SUFFIX.get(name, '')
    data = np.load(f'violin_data/{fname}.npy')
    wboxes = np.load(f'violin_data/weights/{fname}.npy').astype(int)

    cut = np.isnan(data) | np.isinf(data)
    if name == 'bulge_bhmass':
        cut |= data < 5
    if name == 'halo_break':
        cut |= data == 10
    cut |= ~np.isin(wboxes, disk_sample)

    wboxes = wboxes[~cut]
    return data[~cut], weights[np.isin(np.arange(1024), wboxes)]


def table_row(values, w):
    """1st / 99th percentile (unweighted, as violin.py) and the weighted median."""
    s = np.sort(values)
    p1 = s[int(len(s)*0.01)]
    p99 = s[int(len(s)*0.99)]
    wdata = np.repeat(values, (w*10/np.mean(w)).astype(int))
    return p1, np.median(wdata), p99


def main():
    weights = calc_weights()
    disk_sample = DREAMS.get_disk_sample()

    rows = []
    for i, (name, label, units, is_log) in enumerate(PROPS):
        values, w = load_property(name, weights, disk_sample)
        p1, p50, p99 = table_row(values, w)
        rows.append((label, len(values), p1, p50, p99, values.min(), values.mean(), values.max(),
                     OBS_MEAN[i], OBS_LOW[i], OBS_HIGH[i], units, 'Yes' if is_log else 'No'))

    header = (f"{'Property':13s} {'N':>4s} {'1%':>8s} {'50%(w)':>8s} {'99%':>8s} | {'min':>8s} {'mean':>8s} {'max':>8s} |"
              f" {'Observed':>9s} {'Obs. range':>18s}  {'Units':8s} Log")
    print(f"Disk-galaxy sample: {len(disk_sample)} boxes")
    print(header)
    print('-'*len(header))
    for (label, n, p1, p50, p99, mn, me, mx, om, ol, oh, units, lg) in rows:
        print(f"{label:13s} {n:4d} {p1:8.2f} {p50:8.2f} {p99:8.2f} | {mn:8.2f} {me:8.2f} {mx:8.2f} |"
              f" {om:9.2f} [{ol:7.2f}, {oh:7.2f}]  {units:8s} {lg}")

    with open('table2.csv', 'w') as f:
        f.write('property,n,p1,p50_weighted,p99,min,mean,max,obs_mean,obs_low,obs_high,units,log\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print("wrote table2.csv")


if __name__ == '__main__':
    main()
