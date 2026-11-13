import numpy as np
import matplotlib.pyplot as plt

def set_ticks(ax):
    ax.tick_params('both', which='minor', length=4, direction='in', bottom=True, top=True, left=True, right=True)
    ax.tick_params('both', which='major', length=8, direction='in', bottom=True, top=True, left=True, right=True)

    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)

    return

def set_plot_params(nrows=1, ncols=1, figsize=None, **kwargs):

    plt.rc('font',**{'family':'STIXGeneral'})
    plt.rc('text', usetex=True)
    plt.rc('mathtext', fontset='stix')

    plt.rc('font', size=12)          # controls default text sizes
    plt.rc('axes', titlesize=16)     # fontsize of the axes title
    plt.rc('axes', labelsize=24)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    plt.rc('legend', fontsize=16)

    plt.rc('lines', linewidth=5)

    if figsize is None:
        fig, ax = plt.subplots(nrows,ncols, figsize=(ncols*3 + (ncols)*3,nrows*5+(nrows-1)*3))
    else:
        fig, ax = plt.subplots(nrows,ncols, figsize=figsize, **kwargs)

    if type(ax)==type(np.zeros(1)):
        for a in ax.ravel():
            set_ticks(a)
    else:
        set_ticks(ax)

    return fig, ax

def main():

    # Fig E3: contamination-emulator samples with contamination fixed to 0 (generate_data.py,
    # context 'SB'). Columns: 7 conditioning parameters (total_mass, omega_m, sigma_8, log_sn1,
    # log_sn2, log_agn1, contam) followed by the 27 targets in config_flows.py order.
    # Release: the research version read the simulation CSV and had t_gse / t_mm swapped.
    data = np.load("generated_samples/100k_CDM_SB.npy")

    gse_ratio = data[:,9]
    t_gse = data[:,8]
    t_mm = data[:,7]
    f = data[:,11]
    beta = data[:,12]

    cut1 = gse_ratio < 5
    cut2 = (t_gse > 0.33) & (t_gse < 0.5)
    cut3 = t_mm < 0.5
    cut4 = f > 0.5
    cut5 = beta > 0.5

    gse_cut = cut1 & cut2 & cut3 & cut4 & cut5

    print(np.sum(gse_cut), len(gse_cut))

    gse_sfr = data[gse_cut, 17]
    all_sfr = data[~gse_cut, 17]

    #gse_sfr[gse_sfr==0] = 1e-4
    #all_sfr[all_sfr==0] = 1e-4

    #gse_sfr = np.log10(gse_sfr)
    #all_sfr = np.log10(all_sfr)

    bins = np.linspace(-4,2,30)

    fig, ax = set_plot_params()
    ax.hist(all_sfr, bins=bins, color='k', density=True, histtype='step', label='Emulated MW Mass Galaxies', linewidth=3)
    ax.hist(gse_sfr, bins=bins, color='r', density=True, histtype='step', label='Emulated GSE Analogs', linewidth=3)

    ax.set_xlabel("$\log$ SFR [M$_\odot$ yr$^{-1}$]")
    ax.set_ylabel("Normalized Samples per Bin")

    ax.legend(frameon=False, loc='upper right')
    #ax.text(-4, 0.7, , fontsize=14)
    ax.set_title("Fixed to 0\% Contamination")
    ax.set_ylim((0,0.8))

    fig.savefig("plots/gse_sfr.pdf", bbox_inches='tight')

    return

if __name__=='__main__':
    main()
