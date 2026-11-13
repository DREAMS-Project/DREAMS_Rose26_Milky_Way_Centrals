import numpy as np

def main():

    orig = np.loadtxt("gse_prob_cond.csv", delimiter=',', skiprows=1)
    nsat = np.load("nsat.npy")

    these_nsat = nsat[np.newaxis,:].T

    out = np.concatenate((these_nsat,orig),axis=1)

    print(out.shape)

    np.savetxt("new_gse_prob_cond.csv", out, delimiter=',')

    return

if __name__=='__main__':
    main()
