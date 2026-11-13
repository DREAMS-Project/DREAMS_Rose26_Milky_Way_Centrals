import numpy as np
import os
DREAMS_DATA = os.environ.get("DREAMS_DATA", "/path/to/DREAMS")

def main():

    orig = np.loadtxt("gse_prob_cond.csv", delimiter=',', skiprows=1)
    contam = np.load("../../violin_data/halo_contam.npy")

    om = orig[:,2]

    all_om = np.loadtxt(DREAMS_DATA + "/Parameters/CDM/MW_zooms/CDM_TNG_MW_SB5.txt", skiprows=1)[:,0]

    these_boxes = []
    all_boxes = np.arange(1024)
    for i in range(len(om)):
        cut = np.where(all_om==om[i])[0]
        these_boxes.append(all_boxes[cut][0])
    
    these_boxes = np.array(these_boxes)

    these_contam = contam[these_boxes]
    these_contam = these_contam[np.newaxis,:].T

    out = np.concatenate((orig,these_contam),axis=1)

    print(out.shape)

    np.savetxt("new_gse_prob_cond.csv", out, delimiter=',')

    return

if __name__=='__main__':
    main()
