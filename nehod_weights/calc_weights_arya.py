import numpy as np
import plotting
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt

def dist(x, y, lp, hp):
    cross = np.cross([hp[0]-lp[0],hp[1]-lp[1]],[lp[0]-x,lp[1]-y])
    dist = np.linalg.norm(cross)/np.linalg.norm([hp[0]-lp[0],hp[1]-lp[1]])
    return dist

def calc_residuals(data_x, data_y, x, mean):

    binned_residuals = []
    nsamples_per_bin = nspb = 200 # 10 # 200 #per trial
    ntrials = 5
    ncosmo = len(data_x)//(nspb*ntrials) 

    spl = CubicSpline(x, mean)

    print("Starting Weight Calculation")

    for i in range(ncosmo):
        residuals = []
        for j in range(ntrials):
            for k in range(nsamples_per_bin):
                sim_idx = j*ncosmo*nspb + i*nspb + k
                #obs_idx = np.argmin(np.abs(data_x[sim_idx]-x))
                #d = mean[obs_idx] - data_y[sim_idx]
                d = data_y[sim_idx] - spl(data_x[sim_idx])
                residuals.append(d)
        binned_residuals.append(np.mean(residuals))

        if i%1000 == 0:
            print(f"Finished {i} Cosmologies")

    return np.array(binned_residuals)

def get_bins(point, bins, npoints=30):

    all_idx = []
    for i in range(len(bins)):
        idx = np.argmin(np.abs((bins[i] - point[i])))
        all_idx.append(idx)

    npoints = len(bins[0])
    idx = all_idx[0]*npoints*npoints+all_idx[1]*npoints+all_idx[2]

    return idx

def main():

    data = np.load("generated_samples_arya/1000_arya_cdm.npy")

    # UM-SAGA SMHM relation (Wang et al. 2024; obsdata/smhm_{mean,high,low}.csv).
    # This reproduces the arya_weights.npy used in the paper exactly.
    x, mean, high, low = plotting.get_obs('smhm', 11.0, 12.5, 200)

    # Sec. 3.3 sensitivity tests: replace the line above with the block below, using
    # obsdata/Birrer.csv (Birrer et al. 2014) or obsdata/Yang.csv (Yang et al. 2012),
    # which are stored as log(M*/Mh) vs log Mh.
    #obs = np.loadtxt('obsdata/Birrer.csv', delimiter=',')
    #x = obs[:,0]
    #y = obs[:,1]
    #tmpx = np.power(10,x)
    #tmpy = np.power(10,y)
    #mean = np.log10(tmpy * tmpx)

    smass = data[:,5]
    hmass = data[:,6]
    residuals_smhm = calc_residuals(hmass, smass, x, mean)

    #new_likeli = np.load("arya_weights.npy")
    
    for tao_smhm in [0.2]:

        new_likeli_smhm = np.exp(-1*residuals_smhm*residuals_smhm / (2 * tao_smhm * tao_smhm))
        print(new_likeli_smhm.shape)
        np.save("arya_weights", new_likeli_smhm)  # -> reshape_weights.py -> arya_weights_binned.npy

        npoints=30
        sn1 = np.linspace(np.log10(0.9), np.log10(14.4), npoints)
        sn2 = np.linspace(np.log10(3.7), np.log10(14.8), npoints)
        agn = np.linspace(np.log10(0.025), np.log10(0.4), npoints)
        bins = [sn1, sn2, agn]

        print()
        print("Weighting Galaxies")

        all_likeli = []
        for j in range(len(data)):
            idx = get_bins(data[j][2:5], bins)
            all_likeli.append(new_likeli_smhm[idx])
            if j % 1000000 == 0:
                print(f"Weighted {j} Galaxies")
        all_likeli = np.array(all_likeli)
        print("Done!")

        fig, ax = plotting.plot_corner(data, all_likeli, vmin=10**(-2), vmax=1, is_small=True)
        fig.savefig(f"plots/weights_corner.pdf", bbox_inches='tight')
        #fig.savefig(f"plots/tao/tau_{tao_smhm}_weights.pdf", bbox_inches='tight')
        #fig.savefig(f"plots/trials/trial_{i}_new_likeli_best.pdf", bbox_inches='tight')

        x, mean, high, low = plotting.get_obs('smhm', 11.0, 12.5, 200)
        fig, ax = plotting.plot_smhm(data[:,6], data[:,5], all_likeli)
        #fig.savefig(f"plots/tao/tau_{tao_smhm}_smhm.pdf", bbox_inches='tight')
        fig.savefig(f"plots/weights_smhm.pdf", bbox_inches='tight')
        plt.close(fig)

    return

if __name__=='__main__':
    main()
