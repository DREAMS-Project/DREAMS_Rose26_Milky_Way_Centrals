import numpy as np

def main():

    arya_weights = np.load("arya_weights.npy")

    print(arya_weights.shape)

    weights = np.zeros((30,30,30)) - 1
    
    for sn1 in range(30):
        for sn2 in range(30):
            for agn in range(30):

                idx = sn1*900 + sn2*30 + agn
                weights[sn1,sn2,agn] = arya_weights[idx]
    
    print(np.sum(weights == -1))
    print(weights.shape)
    np.save("arya_weights_binned", weights)

    return

if __name__=='__main__':
    main()
