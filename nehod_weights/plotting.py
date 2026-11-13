import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic,binned_statistic_2d
import matplotlib.ticker as mticker
from matplotlib.colors import LogNorm

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

    plt.rc('lines', linewidth=3)

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

def plot_corner(gen_cond, chi2, vmin=0.2, vmax=1, nbins=30, is_small=True, is_log=True):

    if is_small:
        #names = ['SN1', 'SN2', 'BHFF']
        names = [r'$\bar{e}_w$', r'$\kappa_w$', '$\epsilon_{f,\mathrm{high}}$']
        nrows = 3
        size = 13
        fontsize = 35
        gen_cond = gen_cond[:,2:]
    else:
        #names = ['Om', '$\sigma_8$', 'SN1', 'SN2', 'BHFF']
        names = [r'$\Omega_m$', r'$\sigma_8$', r'$\bar{e}_w$', r'$\kappa_w$', r'$\epsilon_{f,\mathrm{high}}$']
        nrows = 5
        size = 25
        fontsize = 40

    bottom_corner = 2 if is_small else 4
        
    #fig, ax = plt.subplots(nrows, nrows, figsize=(size,size))
    fig, ax = set_plot_params(nrows, nrows, figsize=(size,size))
    for i in range(nrows):
        for j in range(nrows):
            if i > j:
                ax[j,i].axis('off')
                continue
            
            if i == j:
                if i > 1 or is_small==True:
                    # Manually define edges to bracket the 30 grid points
                    # This creates 30 bins centered on the 30 points
                    step = (np.max(gen_cond[:,i]) - np.min(gen_cond[:,i])) / (nbins-1)
                    bins = np.logspace(np.min(gen_cond[:,i]) - step/2,np.max(gen_cond[:,i]) + step/2, nbins+1)
                    #ax[i,j].set_xscale("log")
                    x = np.power(10, gen_cond[:,i])
                else:
                    #bins = np.linspace(np.min(gen_cond[:,i]),np.max(gen_cond[:,i]), nbins)

                    step = (np.min(gen_cond[:,i]) - np.max(gen_cond[:,i])) / (nbins-1)
                    bins = np.linspace(np.min(gen_cond[:,i]) - step/2,np.max(gen_cond[:,i]) + step/2, nbins+1)
                    x = gen_cond[:,i]
                    
                if i == 0:
                    ax[i,j].tick_params('x', which='both', labelbottom=False)
                    #ax[i,j].tick_params('y', labelrotation=45)
                elif i == bottom_corner:
                    ax[i,j].tick_params('y', which='both', labelleft=False, labelright=True)
                else:
                    ax[i,j].tick_params('both', which='both', labelleft=False, labelbottom=False, labelright=True)
                
                avg, be, bn = binned_statistic(x, chi2, lambda x:np.quantile(x,0.5), bins=bins)    
                std_low, be, bn = binned_statistic(x, chi2, lambda x:np.quantile(x,0.16), bins=bins)    
                std_high, be, bn = binned_statistic(x, chi2, lambda x:np.quantile(x,0.84), bins=bins)    
    
                bin_centers = (bins[1:] + bins[:-1]) / 2

                ax[i,j].xaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                ax[i,j].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
                ax[i,j].yaxis.set_label_position("right")
                ax[i,j].yaxis.tick_right()
                if is_log:
                    #ax[i,j].set_yscale('log')
                    ax[i,j].yaxis.set_major_formatter(mticker.ScalarFormatter())
                    ax[i,j].ticklabel_format(axis='y',style='plain')
                    ax[i,j].tick_params('y', which='minor', labelleft=False, labelright=True)
                    ax[i,j].plot(np.log10(bin_centers), np.log10(avg), 'k-')
                    ax[i,j].fill_between(np.log10(bin_centers), np.log10(std_high), np.log10(std_low), color='k', alpha=0.5)
                    ax[i,j].set_ylim((np.log10(vmin),np.log10(vmax)))
                else:
                    ax[i,j].plot(bin_centers, avg, 'k-')
                    ax[i,j].fill_between(bin_centers, std_high, std_low, color='k', alpha=0.5)
                    ax[i,j].set_ylim((vmin,vmax))
                    ax[i,j].tick_params('both', labelrotation=45)

                if i > 1 and is_log:
                    xticks = np.linspace(np.log10(bin_centers[0]), np.log10(bin_centers[-1]),5)
                    xtick_labels = np.linspace(np.min(gen_cond[:,i]),np.max(gen_cond[:,i]), 5)
                    ax[j,i].set_xticks(xticks, [f'{np.power(10,el):.2f}' for el in xtick_labels], rotation=45)

                if is_log:
                    ax[i,j].set_ylabel("Log Median $w$", fontsize=fontsize)
                else:
                    ax[i,j].set_ylabel("Avg $w$", fontsize=fontsize)

                continue
            
            step1 = (np.max(gen_cond[:,i]) - np.min(gen_cond[:,i])) / (nbins-1)
            step2 = (np.max(gen_cond[:,j]) - np.min(gen_cond[:,j])) / (nbins-1)
            #bins = np.logspace(np.min(gen_cond[:,i]) - step/2,np.max(gen_cond[:,i]) + step/2, nbins+1)
            bins1 = np.linspace(np.min(gen_cond[:,i])-step1/2, np.max(gen_cond[:,i])+step1/2, nbins+1)
            bins2 = np.linspace(np.min(gen_cond[:,j])-step2/2, np.max(gen_cond[:,j])+step2/2, nbins+1)
            bins = [bins1, bins2]
            
            avg, xbe, ybe, bn = binned_statistic_2d(gen_cond[:,i], gen_cond[:,j], np.log10(chi2), 'median', bins=bins)
            
            bin_centers1 = (bins1[1:] + bins1[:-1]) / 2
            bin_centers2 = (bins2[1:] + bins2[:-1]) / 2
            bin_centers = [bin_centers1, bin_centers2]

            if is_log:
                # ax[j,i].imshow(avg[:,::-1].T, norm=LogNorm(vmin=vmin*10, vmax=vmax*.5), cmap='inferno')
                im = ax[j,i].imshow(avg[:,::-1].T, cmap='inferno', vmin=np.log10(vmin), vmax=np.log10(vmax)) # norm=LogNorm(vmin=vmin, vmax=vmax))
            else:
                im = ax[j,i].imshow(avg[:,::-1].T, vmin=vmin, vmax=vmax, cmap='inferno')
            
            xticks = np.linspace(1,nbins-2,5)
            yticks = np.linspace(1,nbins-2,5)
            xtick_labels = np.linspace(np.min(gen_cond[:,i]),np.max(gen_cond[:,i]), 5)
            ytick_labels = np.linspace(np.min(gen_cond[:,j]),np.max(gen_cond[:,j]), 5)[::-1]
    
            if i > 1:
                ax[j,i].set_xticks(xticks, [f'{np.power(10,el):.2f}' for el in xtick_labels], rotation=45)
            else:
                ax[j,i].set_xticks(xticks, [f'{el:.2f}' for el in xtick_labels], rotation=45)
                
            if j > 1:
                ax[j,i].set_yticks(yticks, [f'{np.power(10,el):.2f}' for el in ytick_labels], rotation=45)
            else:
                ax[j,i].set_yticks(yticks, [f'{el:.2f}' for el in ytick_labels], rotation=45)
            
            if j == bottom_corner:
                ax[j,i].set_xlabel(names[i], fontsize=fontsize)
            else:
                ax[j,i].tick_params('x', which='both', labelbottom=False)
                
            if i == 0:
                ax[j,i].set_ylabel(names[j], fontsize=fontsize)
            else:
                ax[j,i].tick_params('y', which='both', labelleft=False)
            
            ax[bottom_corner,0].tick_params('both', which='both', labelleft=True, labelbottom=True)
                
            if ((j == 4 and not is_small) or (j == 2 and is_small)) and (i==0):
                ax[j,i].tick_params('x', which='both', labelbottom=False)
                ax[j,i].tick_params('y', which='both', labelleft=False)
    
            ax[j,i].plot(nbins//2,nbins//2,'co',markersize=15, linewidth=5, markeredgewidth=4, label='Fiducial TNG Parameters', markerfacecolor='none')

    cax = ax[0,0].inset_axes([1.5,0.4,1.56,0.2])
    cbar = fig.colorbar(im, ax=ax[0,0], cax=cax, orientation='horizontal')
    cbar.set_ticks([-2.0,-1.5,-1.0,-0.5,0.0], labels=['-2.0','-1.5','-1.0','-0.5','0.0'], fontsize=20) 
    cbar.ax.set_xlabel("Log Median $w$", fontsize=fontsize-5) 

    ax[-1,-1].set_xlabel(names[-1], fontsize=fontsize)

    # ax[0,0].set_ylabel(r"Avg $\chi^2$", fontsize=fontsize)

    for i in range(len(ax)):
        for j in range(len(ax[0])):
            
            if i == j:
                ax[i,j].tick_params('both', which='minor', length=4, direction='in', bottom=True, top=False, left=False, right=True)
                ax[i,j].tick_params('both', which='major', length=8, direction='in', bottom=True, top=False, left=False, right=True)

            else:
                ax[i,j].tick_params('both', which='minor', length=4, direction='in', bottom=True, top=False, left=True, right=False)
                ax[i,j].tick_params('both', which='major', length=8, direction='in', bottom=True, top=False, left=True, right=False)

            for axis in ['top','bottom','left','right']:
                ax[i,j].spines[axis].set_linewidth(2)

    #ax[1,0].plot(-10,-10,'+', color='cyan', markersize=15, linewidth=5, markeredgewidth=5)
    ax[1,0].legend(bbox_to_anchor=(3.2,1.9), frameon=False, fontsize=fontsize-5)

    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    return fig, ax

def get_obs(name, mi, ma, npoints=5):

    mean = np.loadtxt(f'obsdata/{name}_mean.csv', delimiter=',') #, skiprows=1)
    high = np.loadtxt(f'obsdata/{name}_high.csv', delimiter=',') #, skiprows=1)
    low = np.loadtxt(f'obsdata/{name}_low.csv', delimiter=',') #, skiprows=1)
    
    x = np.linspace(mi, ma, npoints)
    me = np.interp(x, mean[:,0], mean[:,1])
    hi = np.interp(x, high[:,0], high[:,1])
    lo = np.interp(x, low[:,0], low[:,1])

    return x, me, hi, lo

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    They weights are in effect first normalized so that they 
    sum to 1 (and so they must not all be 0).

    values, weights -- NumPy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)
    return np.sqrt(variance)

def plot_smhm(hmass, smass, weights):

    x, mean, high, low = get_obs('smhm', 11, 12.5, 100)

    fig, ax = set_plot_params(1,3,figsize=(17,5))
    smhm_bins = (np.linspace(np.min(hmass),np.max(hmass),100), np.linspace(np.min(smass),np.max(smass),100))
    wts_cnt, be1, b2, bn = binned_statistic_2d(hmass, smass, weights, 'sum', bins=smhm_bins)
    cnt, be1, b2, bn = binned_statistic_2d(hmass, smass, weights, 'count', bins=smhm_bins)
    #im1 = ax[0].hist2d(hmass, smass, bins=smhm_bins, cmap='inferno', vmin=0, vmax=1500)
    #im2 = ax[1].hist2d(hmass, smass, bins=smhm_bins, cmap='inferno', weights=weights, vmin=0, vmax=400)
    im1 = ax[0].hist2d(hmass, smass, bins=smhm_bins, cmap='Reds', vmin=0, vmax= np.max(cnt)) #25000)
    im2 = ax[1].hist2d(hmass, smass, bins=smhm_bins, cmap='Blues', weights=weights, vmin=0, vmax=np.max(wts_cnt)) #8000)

    text_color = 'black'
    cax = ax[0].inset_axes([0.4,0.12,0.4,0.05])
    cbar1 = fig.colorbar(im1[3], ax=ax[0], cax=cax, orientation='horizontal')
    cbar1.ax.tick_params(axis='x', colors=text_color)
    cbar1.outline.set_edgecolor(text_color)
    #cbar1.set_ticks([0,12500,25000])
    cbar1.ax.set_title("Sample Count")

    cax = ax[1].inset_axes([0.4,0.12,0.4,0.05])
    cbar2 = fig.colorbar(im2[3], ax=ax[0], cax=cax, orientation='horizontal')
    cbar2.ax.tick_params(axis='x', colors=text_color)
    cbar2.outline.set_edgecolor(text_color)
    #cbar2.set_ticks([0,4000,8000])
    cbar2.ax.set_title("Weighted Sample Count")

    ax[0].text(11.6,11.6,"Unweighted",fontsize=20)
    ax[1].text(11.6,11.6,"Weighted",fontsize=20)

    bins = np.linspace(11.5,12.4,15)
    avg_w, be, bn = binned_statistic(hmass, smass*weights, 'mean', bins=bins)
    avg, be, bn = binned_statistic(hmass, smass, 'mean', bins=bins)
    avg_weights, be, bn = binned_statistic(hmass, weights, 'mean', bins=bins)
    std, be, bn = binned_statistic(hmass, smass, 'std', bins=bins)
    stdw, be, bn = binned_statistic(hmass, np.arange(len(hmass)), lambda x:weighted_avg_and_std(smass[x],weights[x]), bins=bins)
    bc = (bins[1:] + bins[:-1]) / 2

    ax[2].plot(bc, avg, 'r-',alpha=0.7, label='Unweighted')
    ax[2].fill_between(bc, avg+std, avg-std, color='r', alpha=0.5)

    ax[2].plot(bc, avg_w/avg_weights, 'b-',alpha=0.7, label='Weighted')
    ax[2].fill_between(bc, avg_w/avg_weights+stdw, avg_w/avg_weights-stdw, color='b', alpha=0.5)

    #ax[2].set_xlim((11.5,12.4))
    #ax[2].set_ylim((8.2,12))

    for axi in ax:
        axi.set_xlabel("log $M_\mathrm{TOT}$ [M$_\odot$]")
        axi.set_ylabel("log $M_*$ [M$_\odot$]")
        axi.plot(x, mean, '--', color='darkgrey', alpha=1)
        axi.plot(x, low, '-', color='darkgrey', alpha=1)
        axi.plot(x, high, '-', color='darkgrey', alpha=1)
        axi.fill_between(x, mean+0.3, mean-0.3, color='darkgrey', alpha=0.5)

    ax[2].plot([0],[0], '--', color='darkgrey', label='UM-SAGA')
    ax[2].legend(frameon=False)
    
    return fig, ax

def plot_msig(vel_disp, bh_mass, weights):

    x, mean, high, low = get_obs('msigma', 0.5, 3.2, 20)

    fig, ax = set_plot_params(1,2,figsize=(10,5))
    msig_bins = (np.linspace(1.3,2.7,100),np.linspace(5,9,100))
    cnt, be1, b2, bn = binned_statistic_2d(vel_disp, bh_mass, weights, 'sum', bins=msig_bins)
    ax[0].hist2d(vel_disp, bh_mass, bins=msig_bins, cmap='inferno', weights=weights, vmin=np.min(cnt), vmax=np.max(cnt))

    bins = bins = np.linspace(1.3,2.7,100)
    #med, be, bn = binned_statistic(hmass, smass, lambda x:np.quantile(x,0.5), bins=bins)
    #std_high, be, bn = binned_statistic(hmass, smass, lambda x:np.quantile(x,0.84), bins=bins)
    #std_low, be, bn = binned_statistic(hmass, smass, lambda x:np.quantile(x,0.16), bins=bins)

    avg_w, be, bn = binned_statistic(vel_disp, bh_mass*weights, 'mean', bins=bins)
    avg, be, bn = binned_statistic(vel_disp, bh_mass, 'mean', bins=bins)
    avg_weights, be, bn = binned_statistic(vel_disp, weights, 'mean', bins=bins)
    std, be, bn = binned_statistic(vel_disp, bh_mass, 'std', bins=bins)
    stdw, be, bn = binned_statistic(vel_disp, np.arange(len(vel_disp)), lambda x:weighted_avg_and_std(bh_mass[x],weights[x]), bins=bins)
    bc = (bins[1:] + bins[:-1]) / 2

    ax[1].plot(bc, avg, 'r-',alpha=0.7)
    ax[1].fill_between(bc, avg+std, avg-std, color='r', alpha=0.5)

    ax[1].plot(bc, avg_w/avg_weights, 'b-',alpha=0.7)
    ax[1].fill_between(bc, avg_w/avg_weights+stdw, avg_w/avg_weights-stdw, color='b', alpha=0.5)

    print(np.mean(stdw*2))

    #ax[1].set_xlim((11.5,12.5))
    #ax[1].set_ylim((8.2,12))

    ax[1].set_ylim((5,9))
    ax[1].set_xlim((1.7,2.7))

    for axi in ax:
        plt.xlabel("log $\sigma_x$ [km s$^{-1}$]")
        plt.ylabel("M$_\mathrm{BH}$ [M$_\odot$]")
        axi.plot(x, mean, '--', color='darkgrey', alpha=1)
        axi.plot(x, low, '-', color='darkgrey', alpha=1)
        axi.plot(x, high, '-', color='darkgrey', alpha=1)
    
    return fig, ax
