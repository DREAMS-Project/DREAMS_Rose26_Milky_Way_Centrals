import sys
import matplotlib.pyplot as plt
import  numpy as np
import shared_data as sd

# Compares the scatter in disk-selected TNG50/TNG100 MW-mass centrals to the DREAMS flow
# sampled at fiducial TNG parameters and the same M200c distribution (generate_tng.py).

mfof_min, mfof_max = 5e11, 2.5e12   #DREAMS.get_MW_idx selection
sfr_floor = -3.9                    #log SFR at/below this is the 1e-4 floor -> quenched
n_boot = 1000

def width(x):
    p16, p84 = np.percentile(x, [16,84])
    return (p84 - p16) / 2

def fixed_mass_scatter(x, logm):
    #scatter about a linear median-ish relation with halo mass
    coef = np.polyfit(logm, x, 1)
    return width(x - np.polyval(coef, logm))

def boot(stat, rng, *arrs):
    n = len(arrs[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(stat(*[a[idx] for a in arrs]))
    return np.std(vals)

def get_stats(x, logm, rng, bootstrap):
    out = {'N': len(x), 'median': np.median(x), 'width': width(x), 'width_m': fixed_mass_scatter(x, logm)}
    if bootstrap:
        out['median_err'] = boot(np.median, rng, x)
        out['width_err'] = boot(width, rng, x)
        out['width_m_err'] = boot(fixed_mass_scatter, rng, x, logm)
    return out

def main():

    #(TNG key, flow/DREAMS key, label); TNG keys come from catalog_{box} and stellar_halo_{box}
    props = [('smass', 'halo_tot_smass', r'$\log M_*$'),
             ('bmass', 'halo_bmass', r'$\log M_\mathrm{bary}$'),
             ('sfr', 'halo_sfr', r'$\log \mathrm{SFR}$')]
    props_extra = [('disk_size', 'disk_size', r'$R_\mathrm{disk}\ [\mathrm{kpc}]$'),
                   ('halo_central_smass', 'halo_central_smass', r'$\log M_{*,\mathrm{halo}}$'),
                   ('halo_break', 'halo_break', r'$r_s\ [\mathrm{kpc}]$'),
                   ('halo_q_in', 'halo_q_in', r'$q_\mathrm{in}$'),
                   ('halo_q_out', 'halo_q_out', r'$q_\mathrm{out}$')]

    props = [('smass', 'halo_tot_smass', r'$\log M_*\ [\mathrm{M}_\odot]$'),
             ('sfr', 'halo_sfr', r'$\log \mathrm{SFR}\ [\mathrm{M}_\odot\,\mathrm{yr}^{-1}]$')]
    props_extra = [('halo_central_smass', 'halo_central_smass', r'$\log M_{*,\mathrm{halo}}\ [\mathrm{M}_\odot]$'),
                   ('halo_q_in', 'halo_q_in', r'$q_\mathrm{in}$')]

    args = sys.argv[1:]
    tag = ''
    if '--all' in args:
        args.remove('--all')
        props += props_extra
        tag = '_all'
    names, fnames, labels = [list(p) for p in zip(*props)]
    tngs = args if len(args) > 0 else ['tng50', 'tng100']
    tng_labels = [t.upper() for t in tngs]

    rng = np.random.default_rng(13)
    dreams = np.loadtxt("../../nehod_contam/data/gse_prob_cond.csv", delimiter=',', skiprows=1)
    dreams_cols = open("../../nehod_contam/data/gse_prob_cond.csv").readline().strip().split(',')

    rows = []
    tng_hist = [{} for _ in names]   #per property: {box label: TNG data}
    flow_hist = [[] for _ in names]  #per property: flow samples from each box, pooled for plotting
    for j, tng in enumerate(tngs):

        cat = dict(np.load(f"np_arrays/catalog_{tng}.npz"))
        cat['disk_size'] = np.where(cat['disk_fit_ok'], cat['disk_size'], np.nan)
        if tag:
            halo = np.load(f"np_arrays/stellar_halo_{tng}.npz")
            hidx = np.searchsorted(halo['sub_id'], cat['sub_id'])
            hidx[hidx == len(halo['sub_id'])] = 0
            matched = halo['sub_id'][hidx] == cat['sub_id']
            for n in names:
                if n in halo:
                    cat[n] = np.where(matched, halo[n][hidx], np.nan)
        mfof = np.power(10, cat['log_mfof'])
        sel = cat['has_disk'] & (mfof > mfof_min) & (mfof < mfof_max)
        logm = cat['log_m200c'][sel]

        flow = np.load(f"generated_data/flow_{tng}_TNG.npz")
        flow_names = list(flow['target_names'])
        flow_logm = flow['context'][:,0]
        assert np.array_equal(flow['sub_id'], cat['sub_id'][sel])

        print(f"{tng}: {np.sum(sel)} disk galaxies, {len(flow_logm)} flow samples")

        for i in range(len(names)):

            tng_data = cat[names[i]][sel]
            ok = np.isfinite(tng_data)
            if np.sum(~ok) > 0:
                print(f"  {names[i]}: dropping {np.sum(~ok)} TNG galaxies with failed measurements")
            tng_data, tng_logm = tng_data[ok], logm[ok]
            flow_data = flow['samples'][:,flow_names.index(fnames[i])]
            dreams_data = dreams[:,dreams_cols.index(fnames[i])]

            data = {'TNG': (tng_data, tng_logm), 'flow': (flow_data, flow_logm)}
            for src, (x, m) in data.items():
                row = {'box': tng, 'prop': names[i], 'sample': src}
                if names[i] == 'sfr':
                    sf = x > sfr_floor
                    row['quenched_frac'] = 1 - np.mean(sf)
                    x, m = x[sf], m[sf]
                row.update(get_stats(x, m, rng, bootstrap=(src=='TNG')))
                rows.append(row)

            tng_hist[i][tng_labels[j]] = tng_data
            flow_hist[i].append(flow_data)

    #one row: DREAMS (flow at TNG physics, pooled over all boxes' halo masses) vs each TNG box
    colors = ['r', 'navy', 'tab:green']
    fig, ax = sd.set_plot_params(1, len(names), figsize=(len(names)*6, 5))
    ax = np.atleast_1d(ax)
    for i in range(len(names)):
        flow_data = np.concatenate(flow_hist[i])
        allx = np.concatenate([flow_data] + list(tng_hist[i].values()))
        lo, hi = np.percentile(allx, [0.5, 99.5])
        bins = np.linspace(lo, hi, 30)

        a = ax[i]
        a.hist(flow_data, bins=bins, density=True, color='0.75', label='DREAMS (TNG physics)')
        for k, (lab, x) in enumerate(tng_hist[i].items()):
            a.hist(x, bins=bins, density=True, histtype='step', color=colors[k], label=lab, linewidth=2)
        a.set_xlabel(labels[i])

    ax[0].set_ylabel("PDF")
    ax[0].set_ylim((0,2))
    handles, leg_labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, leg_labels, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 0.89), ncol=len(handles))
    fig.savefig(f"plots/comparison_disk_emulated_{'_'.join(tngs)}{tag}.pdf", bbox_inches='tight')

    keys = ['box', 'prop', 'sample', 'N', 'median', 'median_err', 'width', 'width_err', 'width_m', 'width_m_err', 'quenched_frac']
    with open(f"plots/comparison_disk_stats_{'_'.join(tngs)}{tag}.csv", 'w') as f:
        f.write(','.join(keys) + '\n')
        for row in rows:
            f.write(','.join(str(row.get(k, '')) for k in keys) + '\n')

    print(f"{'box':7s}{'prop':20s}{'sample':6s}{'N':>8s}{'median':>14s}{'width':>14s}{'width|M':>14s}{'f_q':>7s}")
    for r in rows:
        fmt = lambda k: f"{r[k]:6.3f}" + (f"±{r[k+'_err']:.3f}" if k+'_err' in r else '      ')
        fq = f"{r['quenched_frac']:7.3f}" if 'quenched_frac' in r else ''
        print(f"{r['box']:7s}{r['prop']:20s}{r['sample']:6s}{r['N']:8d}{fmt('median'):>14s}{fmt('width'):>14s}{fmt('width_m'):>14s}{fq}")

    return

if __name__=='__main__':
    main()
