import numpy as np
import os
import DREAMS
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def normalize_data(data, obs_mean, obs_high, obs_low, weights=None):

    if weights is not None:
        repeats = weights*10/np.mean(weights)
        wdata = np.repeat(data, repeats.astype(int))
    else:
        wdata = data

    #remove sfr=0 galaxies
    if obs_mean == 0.2355:
        mu = np.mean(wdata[wdata!=-4])
    else:
        #mu = np.mean(wdata)
        mu = np.median(wdata)

    mi = np.min(wdata)
    ma = np.max(wdata)

    sdata = np.sort(data)
    mi = sdata[int(len(sdata)*0.01)]
    ma = sdata[int(len(sdata)*0.99)]

    wdata = np.clip(wdata,mi,ma)

    data = np.array(list(wdata) + [obs_low,obs_mean,obs_high])

    if False:
        sigmas = [0.682, 0.954, 0.9973, 0.999936]

        med = np.quantile(data, 0.5)
        lows = []
        highs = []
        for si in sigmas:
            lows.append(np.quantile(data, 0.5 - si/2))
            highs.append(np.quantile(data, si/2 + 0.5))

        y = np.arange(-4,5)
        x = lows + [med] + highs

        norm_data = np.interp(data, x, y)

    if False:
        mu = np.mean(data)
        low_cut = data < mu
        data[low_cut] = (data[low_cut] - mu) / sigma_low
        data[~low_cut] = (data[~low_cut] - mu) / sigma_high

    low_cut = data < mu
    data[low_cut] = (mu - data[low_cut]) / (mi - mu)
    data[~low_cut] = (data[~low_cut] - mu) / (ma - mu)

    return data[:-3], data[-3:]

def get_bin_idx(point, npoints=30):

    sn1 = np.linspace(np.log10(0.9), np.log10(14.4), npoints)
    sn2 = np.linspace(np.log10(3.7), np.log10(14.8), npoints)
    agn = np.linspace(np.log10(0.025), np.log10(0.4), npoints)

    bins = [sn1, sn2, agn]
    all_idx = []
    for i in range(len(bins)):
        idx = np.argmin(np.abs((bins[i] - point[i])))
        all_idx.append(idx)

    return np.array(all_idx)

def calc_weights():

    #binned_weights = np.load("binned_weights.npy")
    binned_weights = np.load("nehod_weights/arya_weights_binned.npy")
    params = np.loadtxt(DREAMS_DATA + '/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt', skiprows=1)

    fiducial = np.array([3.6,7.4,0.1])

    weights = []
    for i in range(1024):
        sat_param = np.log10(params[i][2:] * fiducial)
        idx = get_bin_idx(sat_param)
        sat_weight = binned_weights[idx[0]][idx[1]][idx[2]]
        weights.append(sat_weight)
    weights = np.array(weights)

    return weights

def main():

    fig, ax = DREAMS.set_plot_params(figsize=(15,7))

    #observed_mean  = [10.699, 9.431, 10.9294, -1.155, .2355, 0.70, 10.265, 10.190, 6.623, .175, 113, 238, 10.613, 2.51, 0.387, 8.740, 25, -2.5, -4.35, 0.65, 0.8]
    #observed_high  = [10.778, 9.544, 10.9918, -1.093, .2865, 0.77, 10.281, 10.230, 6.643, .100, 116, 253, 10.711, 2.97, 0.457, 8.845, 35, -2.2, -3.70, 0.70, 0.9]
    #observed_low   = [10.602, 9.279, 10.8670, -1.217, .1845, 0.63, 10.248, 10.146, 6.602, .250, 110, 223, 10.485, 2.06, 0.319, 8.602, 15, -2.8, -5.00, 0.60, 0.7]

    #observed_mean  = [10.699, 9.431, 10.9294, 0.2175, 0.70, 10.265, 10.190, 6.623, 113, 238, 10.613, 2.51, 0.387, 8.740, 25, -2.5, -4.35, 0.65, 0.8]
    #observed_high  = [10.778, 9.544, 10.9918,  0.265, 0.77, 10.281, 10.230, 6.643, 116, 253, 10.711, 2.97, 0.457, 8.845, 35, -2.2, -3.70, 0.70, 0.9]
    #observed_low   = [10.602, 9.279, 10.8670, 0.1644, 0.63, 10.248, 10.146, 6.602, 110, 223, 10.485, 2.06, 0.319, 8.602, 15, -2.8, -5.00, 0.60, 0.7]
    observed_mean  = [10.699, 9.431, 10.9294, 0.2175, 0.70, 10.265, 10.190, 6.623, 113, 238, 10.613, 2.51, 8.740, 25, -2.5, -4.35, 0.65, 0.8]
    observed_high  = [10.778, 9.544, 10.9918,  0.265, 0.77, 10.281, 10.230, 6.643, 116, 253, 10.711, 2.97, 8.845, 35, -2.2, -3.70, 0.70, 0.9]
    observed_low   = [10.602, 9.279, 10.8670, 0.1644, 0.63, 10.248, 10.146, 6.602, 110, 223, 10.485, 2.06, 8.602, 15, -2.8, -5.00, 0.60, 0.7]

    #for i in range(len(observed_mean)):
    #    print(f'{observed_mean[i]:.2f}, {observed_low[i]:.2f}, {observed_high[i]:.2f}')
    #return

    full_data = []
    #names = ['halo_tot_smass','halo_sat_smass', 'halo_bmass', 'halo_bmass_frac', 'halo_sfr']
    names = ['halo_tot_smass','halo_sat_smass', 'halo_bmass', 'halo_sfr']
    ngal = len(names)
    #names += ['bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'bulge_dm_frac', 'bulge_vdisp']
    names += ['bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'bulge_vdisp']
    nbulge = len(names) - ngal
    names += ['disk_vcirc', 'disk_smass', 'disk_size'] #, 'disk_height']
    ndisk = len(names) - nbulge - ngal
    names += ['halo_central_smass',  'halo_break', 'halo_slope_in', 'halo_slope_out', 'halo_q_in', 'halo_q_out']
    nhalo = len(names) - nbulge - ndisk - ngal

    #labels = ['$\mathrm{M_*}$', '$\mathrm{M_{*,sat}}$', '$\mathrm{M_{bary}}$', '$\mathrm{M_{bary}}$ / $\mathrm{M_{TOT}}$', 'SFR']
    labels = ['$\mathrm{M_*}$', '$\mathrm{M_{*,sat}}$', '$\mathrm{M_{bary}}$', 'SFR']
    #labels += ['$\mathrm{R_{eff}}$', '$\mathrm{M_{TOT,bulge}}$', '$\mathrm{M_{*,bulge}}$', '$\mathrm{M_{BH}}$', '$\mathrm{M_{*,bulge}}$/$\mathrm{M_{TOT,bulge}}$', '$\sigma_\mathrm{*,bulge}$']
    labels += ['$\mathrm{R_{eff}}$', '$\mathrm{M_{TOT,bulge}}$', '$\mathrm{M_{*,bulge}}$', '$\mathrm{M_{BH}}$', '$\sigma_\mathrm{*,bulge}$']
    labels += ['V$_\mathrm{circ}$', '$\mathrm{M_{*,disk}}$', '$\mathrm{R_{disk}}$'] #, '$z_\mathrm{disk}$']
    labels += ['$\mathrm{M_{*,halo}}$', '$r_s$', r'$\alpha_\mathrm{in}$', r'$\alpha_\mathrm{out}$', "q$_\mathrm{in}$", "q$_\mathrm{out}$"]

    # Release: Fig 3 uses the stochastic-circularity decomposition (Appendix C). bulge_mass.py and
    # disk_size.py write those outputs with an '_old' suffix; the un-suffixed bulge/disk files are the
    # Zana et al. (2022) comparison (disk_size_new.py), used only for the Sec. 4.1 robustness check.
    file_suffix = {n: '_old' for n in ['bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'disk_smass', 'disk_size']}
    disk_sample = DREAMS.get_disk_sample()

    colors = ['k']*ngal + ['b']*nbulge + ['orange']*ndisk + ['m']*nhalo

    weights = calc_weights()

    for i,name in enumerate(names):
        fname = name + file_suffix.get(name, '')
        this_data = np.load(f'violin_data/{fname}.npy')
        wpath = f'violin_data/weights/{fname}.npy'

        if os.path.exists(wpath):
            wboxes = np.load(wpath).astype(int)
            nan_cut = np.isnan(wboxes) | np.isnan(this_data) | np.isinf(this_data)

            if 'bulge_bhmass' == name:
                nan_cut = nan_cut | (this_data < 5)
            if 'halo_break' == name:
                nan_cut = nan_cut | (this_data == 10)

            nan_cut = nan_cut | ~np.isin(wboxes, disk_sample)  # restrict to the disk-galaxy sample
            wboxes = wboxes[~nan_cut]
            wcut = np.isin(np.arange(1024),wboxes)
            this_weights = weights[wcut]
            good_data = this_data[~nan_cut]
        else:
            this_weights = None
            nan_cut = np.isnan(this_data)
            good_data = this_data[~nan_cut]

        fig1,ax1 = DREAMS.set_plot_params()
        if i == 4:
            bins = np.linspace(0.2,4,50)
            ax1.set_ylim((0,205))
        elif i == 5:
            bins = np.linspace(8.9,11, 50)
            ax1.set_ylim((0,46))
        elif i == 6:
            bins = np.linspace(8.1,11,50)
            ax1.set_ylim((0,49))
        elif i == 10:
            bins = np.linspace(8,11.1,50)
            ax1.set_ylim((0,59))
        elif i == 11:
            bins = np.linspace(1.2,15,50)
            ax1.set_ylim((0,56))
        else:
            bins = 50
        
        n, b, p = ax1.hist(good_data, bins=bins)
        ax1.fill_betweenx([0,np.max(n)], [observed_low[i], observed_low[i]],[observed_high[i], observed_high[i]], color='k', alpha=0.4)
        fig1.savefig(f"plots/violin/{name}.pdf", bbox_inches='tight')    

        print(f"{name}, {np.min(good_data):.2f}, {np.mean(good_data):.2f}, {np.max(good_data):.2f}, {len(good_data)}")
        norm_data, obs_data = normalize_data(good_data, observed_mean[i], observed_high[i], observed_low[i], this_weights)
        full_data.append(norm_data)

        if obs_data[0] > obs_data[1]:
            obs_data = obs_data[::-1]

        #frac = np.sum((norm_data < obs_data[2]) & (norm_data > obs_data[0])) / len(norm_data)
        #ax.text(i+1-0.3, 1.3, f'{frac*100:.1f}%', fontsize=12, color=colors[i])

        if i == 0:
            ax.errorbar(i+1,obs_data[1],yerr=[[obs_data[1]-obs_data[0]],[obs_data[2]-obs_data[1]]], fmt='ko', capsize=7, linewidth=1, markersize=3, label='Observed Milky Way')
        else:
            ax.errorbar(i+1,obs_data[1],yerr=[[obs_data[1]-obs_data[0]],[obs_data[2]-obs_data[1]]], fmt='ko', capsize=7, linewidth=1, markersize=3)
    

    violin = ax.violinplot(full_data, showextrema=False)

    for i,v in enumerate(violin['bodies']):
        v.set_facecolor(colors[i])
    
    ax.plot([ngal+.5]*2,[-1,5], 'k--', alpha=0.5)
    ax.plot([nbulge+ngal+.5]*2,[-1,5], 'k--', alpha=0.5)
    ax.plot([nbulge+ndisk+ngal+.5]*2,[-1,5], 'k--', alpha=0.5)

    ax.set_xticks(np.arange(len(labels))+1,labels,rotation=45, ha='right', fontsize=24)

    ax.set_ylim((-1.3,1.3))
    ax.set_xlim((0.5,ngal+nbulge+ndisk+nhalo+0.5))
    ax.set_ylabel("Normalized MW Properties", fontsize=32)

    ax.set_yticks([-1,0,1],['1%','50%','99%'], fontsize=24)
    ax.text(1, 1.1, 'Galaxy', fontsize=18)
    ax.text(ngal+1, 1.1, 'Bulge', fontsize=18)
    ax.text(ngal+nbulge+1, 1.1, 'Disk', fontsize=18)
    ax.text(ngal+nbulge+ndisk+1, 1.1, 'Halo', fontsize=18)

    leg1 = ax.legend([violin['bodies'][0]], ['DREAMS Galaxies'], loc='lower left', frameon=False)
    leg2 = ax.legend(loc='lower right', frameon=False)
    ax.add_artist(leg1)

    fig.savefig("plots/violin.png", bbox_inches='tight', dpi=400)
    fig.savefig("plots/violin.pdf", bbox_inches='tight')

    return

if __name__=='__main__':
    main()
