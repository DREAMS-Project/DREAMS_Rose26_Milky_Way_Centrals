import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic as bs

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



def main(show_sim=False):
    cv = np.load("generated_samples/100k_CDM_CV.npy")
    cont = np.load("generated_samples/100k_CDM_cont.npy")

    # Implement sharex=True to combine x-axes and reduce figsize for a two-column publication layout
    fig, axes = set_plot_params(5, 5, figsize=(25, 16), sharex=True)
    
    # Remove vertical white space between plots
    fig.subplots_adjust(hspace=0.05, wspace=0.35)
    
    ax = axes.ravel()

    titles = [
        'total_mass', "omega_m", "sigma_8", "log_sn1", "log_sn2", "log_agn1", "contam",
        'mm_time', 'gse_time', 'gse_ratio', 'ngse', 'halo_f','halo_beta',
        'halo_tot_smass','halo_sat_smass', 'halo_bmass', 'halo_bmass_frac', 'halo_sfr',
        'bulge_reff', 'bulge_mass', 'bulge_smass', 'bulge_bhmass', 'bulge_dm_frac', 'bulge_vdisp',
        'disk_vcirc', 'disk_smass', 'disk_size', 'disk_height',
        'halo_central_smass', 'halo_break', 'halo_slope_in', 'halo_slope_out', 'halo_q_in', 'halo_q_out'
    ]

    sim_data = np.loadtxt("data/gse_prob_cond.csv", delimiter=',', skiprows=1)

    contam = sim_data[:,-1]

    labels = [r'$t_{\rm mm}$', r'$t_{\rm gse}$', r'$M_{\rm gse} / M_{\rm MW}$', r'N$_{\rm gse}$', '$f$', r'$\beta$']
    labels += ['$\mathrm{M_*}$', '$\mathrm{M_{*,sat}}$', '$\mathrm{M_{bary}}$', 'SFR']
    labels += ['$\mathrm{R_{eff}}$', '$\mathrm{M_{TOT,bulge}}$', '$\mathrm{M_{*,bulge}}$', '$\mathrm{M_{BH}}$', '$\sigma_\mathrm{*,bulge}$']
    labels += ['V$_\mathrm{circ}$', '$\mathrm{M_{*,disk}}$', '$\mathrm{R_{disk}}$']
    labels += ['$\mathrm{M_{*,halo}}$', '$r_s$', r'$\alpha_\mathrm{in}$', r'$\alpha_\mathrm{out}$', "q$_\mathrm{in}$", "q$_\mathrm{out}$"]

    offset = 0
    for i, prop in enumerate(range(7, 34)):

        i -= offset

        if prop in [16,22,27]:
            offset += 1
            continue


        if show_sim:
            sim_plot = ax[i].plot(contam, sim_data[:,prop], 'r+', alpha=0.5)

        cv_prop = cv[:, prop]
        cont_prop = cont[:, prop]

        bins = np.linspace(0, 0.02, 20)
        bc = (bins[1:] + bins[:-1]) / 2

        mean, be, bn = bs(cont[:, 6], cont_prop, 'mean', bins=bins)
        std, be, bn = bs(cont[:, 6], cont_prop, 'std', bins=bins)

        mean_plot = ax[i].plot(bc, mean, 'k', alpha=0.7)
        ax[i].fill_between(bc, mean+std, mean-std, color='k', alpha=0.4)

        if labels[i] == 'SFR':
            print(bc)
            print(mean)

        ax[i].set_ylabel(labels[i])
        
        # Apply x-label only to the lowest visible subplot in each column
        if i in [19,20,21,22,23]:
            ax[i].set_xlabel("Contamination")
            ax[i].tick_params(labelbottom=True)


    # Disable the axis for the remaining empty subplot (index 27)
    for j in range(i + 1, len(ax)):
        ax[j].axis('off')

    # Populate the legend in the final empty subplot space
    handles, names = [mean_plot[0]], ['Emulated Data']
    if show_sim:
        handles.insert(0, sim_plot[0])
        names.insert(0, 'Simulation Data')

    ax[-1].legend(
        handles,
        names,
        loc='center', 
        fontsize=18, 
        frameon=False
    )

    outname = "plots/property_contamination.pdf" if show_sim else "plots/property_contamination_nosim.pdf"
    fig.savefig(outname, bbox_inches='tight')

    return

if __name__ == '__main__':
    main()
