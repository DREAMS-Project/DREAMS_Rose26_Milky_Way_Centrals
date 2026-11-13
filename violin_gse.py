import numpy as np
import os
import DREAMS
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def normalize_data(data, obs_mean, obs_high, obs_low, weights=None,mi=None,mu=None,ma=None):

    if weights is not None:
        repeats = weights*10/np.mean(weights)
        wdata = np.repeat(data, repeats.astype(int))
    else:
        wdata = data

    if mu is None:
        #remove sfr=0 galaxies
        if obs_mean == 0.2355:
            mu = np.mean(wdata[wdata!=-4])
        else:
            #mu = np.mean(wdata)
            mu = np.median(wdata)

    sdata = np.sort(data)
    if mi is None:
        #mi = np.min(wdata)
        mi = sdata[int(len(sdata)*0.01)]
    if ma is None:
        #ma = np.max(wdata)
        ma = sdata[int(len(sdata)*0.99)]

    wdata = np.clip(wdata,mi,ma)

    data = np.array(list(wdata) + [obs_low,obs_mean,obs_high])

    low_cut = data < mu
    data[low_cut] = (mu - data[low_cut]) / (mi - mu)
    data[~low_cut] = (data[~low_cut] - mu) / (ma - mu)

    return data[:-3], data[-3:], mi, mu, ma

def get_bin_idx(point, npoints=30):  # Release: 30 to match arya_weights_binned.npy (30^3); was 29

    om = np.linspace(0.274, 0.354, npoints)
    s8 = np.linspace(0.780, 0.888, npoints)
    sn1 = np.linspace(np.log10(0.9), np.log10(14.4), npoints)
    sn2 = np.linspace(np.log10(3.7), np.log10(14.8), npoints)
    agn = np.linspace(np.log10(0.025), np.log10(0.4), npoints)

    bins = [om, s8, sn1, sn2, agn]
    bins = [sn1, sn2, agn]
    all_idx = []
    for i in range(len(bins)):
        idx = np.argmin(np.abs((bins[i] - point[i])))
        all_idx.append(idx)

    return np.array(all_idx)

def calc_weights(params=None):

    #binned_weights = np.load("binned_weights.npy")
    binned_weights = np.load("nehod_weights/arya_weights_binned.npy")
    if params is None:
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

    colors = ['k']*ngal + ['b']*nbulge + ['orange']*ndisk + ['m']*nhalo

    gse_data = np.load("gse_data.npy").T
    gse_weights = np.load("gse_weights.npy").T
    print(gse_weights[0])

    #gse_data = np.load("TNG_data.npy").T
    #gse_weights = np.load("TNG_weights.npy")

    full_data = np.load("full_data.npy").T
    full_weights = np.load("full_weights.npy").T
    print(full_weights[0])

    print(gse_data.shape, gse_weights.shape)
    print(full_data.shape, full_weights.shape)

    #good_cols = 12+np.array([0,1,2,4,5,6,7,8,10,11,12,13,14,15,16,17,18,19,20])
    #gse_data = gse_data[good_cols].T
    #gse_weights = gse_weights[good_cols].T


    weights = calc_weights()

    shuffle_idx = np.arange(len(gse_weights))
    np.random.default_rng(13).shuffle(shuffle_idx)  # Release: seeded subsample
    shuffle_idx = shuffle_idx[:800]

    # Release: full_data.npy / gse_data.npy have 19 columns (index 12 is z_disk, which is not
    # plotted) while `names` has 18 entries; the research code indexed column i directly, so every
    # halo violin read the column to its left. Map name index -> data column explicitly.
    data_cols = list(range(12)) + list(range(13, 19))

    # Release: normalise against the same simulation files and disk sample as violin.py (Fig 3)
    file_suffix = {n: '_old' for n in ['bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'disk_smass', 'disk_size']}
    disk_sample = DREAMS.get_disk_sample()

    left_data = []
    right_data = []
    for i,name in enumerate(names):


        if True:
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
                wcut = np.in1d(np.arange(1024),wboxes)
                this_weights = weights[wcut]
                good_data = this_data[~nan_cut]
            else:
                this_weights = None
                nan_cut = np.isnan(this_data)
                good_data = this_data[~nan_cut]

            norm_data, obs_data, mi, mu, ma = normalize_data(good_data, observed_mean[i], observed_high[i], observed_low[i], this_weights)

        if True:
            good_data = full_data[data_cols[i]][shuffle_idx]
            this_weights = full_weights[data_cols[i]][shuffle_idx]
            #this_weights = None

            if 'bulge_bhmass' == name:
                nan_cut = good_data < 5
                good_data = good_data[~nan_cut]
                this_weights = this_weights[~nan_cut]
            if 'halo_break' == name:
                nan_cut = (good_data > 9) & (good_data < 11)
                good_data = good_data[~nan_cut]
                this_weights = this_weights[~nan_cut]

            #print(f"{name}, {np.min(good_data):.2f}, {np.mean(good_data):.2f}, {np.max(good_data):.2f}, {len(good_data)}")
            norm_data, obs_data, mi, mu, ma = normalize_data(good_data, observed_mean[i], observed_high[i], observed_low[i], this_weights, ma=ma,mu=mu,mi=mi)
        left_data.append(norm_data)

        if obs_data[0] > obs_data[1]:
            obs_data = obs_data[::-1]

        frac = np.sum((norm_data < obs_data[2]) & (norm_data > obs_data[0])) / len(norm_data)
        ax.text(i+1-0.3, 1.3, f'{frac*100:.1f}%', fontsize=12, color=colors[i])

        if i == 0:
            ax.errorbar(i+1,obs_data[1],yerr=[[obs_data[1]-obs_data[0]],[obs_data[2]-obs_data[1]]], fmt='ko', capsize=7, linewidth=1, markersize=3, label='Observed Milky Way')
        else:
            ax.errorbar(i+1,obs_data[1],yerr=[[obs_data[1]-obs_data[0]],[obs_data[2]-obs_data[1]]], fmt='ko', capsize=7, linewidth=1, markersize=3)

        this_gse = gse_data[data_cols[i]][shuffle_idx]
        this_gse_weights = gse_weights[data_cols[i]][shuffle_idx]
        #this_gse_weights = None

        print(name, np.min(this_gse), np.max(this_gse), len(this_gse))

        if 'bulge_bhmass' == name:
            nan_cut = this_gse < 5
            this_gse = this_gse[~nan_cut]
            this_gse_weights = this_gse_weights[~nan_cut]
        if 'halo_break' == name:
            nan_cut = (this_gse > 9) & (this_gse < 11)
            this_gse = this_gse[~nan_cut]
            this_gse_weights = this_gse_weights[~nan_cut]

        if False:
            fig, ax = DREAMS.set_plot_params()
            ax.hist(good_data, bins=20, alpha=0.5)
            ax.hist(this_gse, bins=20, alpha=0.5)
            fig.savefig("plots/gse_test.pdf", bbox_inches='tight')
            return

        norm_data, obs_data, ma, mu, mi = normalize_data(this_gse, observed_mean[i], observed_high[i], observed_low[i], this_gse_weights, ma=ma,mu=mu,mi=mi)
        right_data.append(norm_data)       

        if obs_data[0] > obs_data[1]:
            obs_data = obs_data[::-1]

        frac = np.sum((norm_data < obs_data[2]) & (norm_data > obs_data[0])) / len(norm_data)
        ax.text(i+1-.3, 1.2, f'{frac*100:.1f}%', fontsize=12, color='darkgreen')


    left_violin = ax.violinplot(left_data, showextrema=False, side='low')
    #print(right_data)
    right_violin = ax.violinplot(right_data, showextrema=False, side='high')

    for i,v in enumerate(left_violin['bodies']):
        v.set_facecolor(colors[i])
    for i,v in enumerate(right_violin['bodies']):
        v.set_facecolor('darkgreen')
    
    ax.plot([ngal+.5]*2,[-1,5], 'k--', alpha=0.5)
    ax.plot([nbulge+ngal+.5]*2,[-1,5], 'k--', alpha=0.5)
    ax.plot([nbulge+ndisk+ngal+.5]*2,[-1,5], 'k--', alpha=0.5)

    ax.set_xticks(np.arange(len(labels))+1,labels,rotation=45, ha='right', fontsize=24)

    ax.set_ylim((-1.3,1.6))
    ax.set_xlim((0.5,ngal+nbulge+ndisk+nhalo+0.5))
    ax.set_ylabel("Normalized MW Properties", fontsize=32)

    ax.set_yticks([-1,0,1],['1%','50%','99%'], fontsize=24)
    ax.text(1, 1.4, 'Galaxy', fontsize=18)
    ax.text(ngal+1, 1.4, 'Bulge', fontsize=18)
    ax.text(ngal+nbulge+1, 1.4, 'Disk', fontsize=18)
    ax.text(ngal+nbulge+ndisk+1, 1.4, 'Halo', fontsize=18)

    leg1 = ax.legend([left_violin['bodies'][0],right_violin['bodies'][0]], ['DREAMS Galaxies (Emulated)', 'GSE Analogs (Emulated)'], loc='lower left', frameon=False, ncols=2)
    leg2 = ax.legend(loc='lower right', frameon=False)
    ax.add_artist(leg1)

    fig.savefig("plots/violin_GSE.png", bbox_inches='tight', dpi=200)
    fig.savefig("plots/violin_GSE.pdf", bbox_inches='tight')

    return

if __name__=='__main__':
    main()
