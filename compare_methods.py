import DREAMS
import os
import numpy as np

def main():

    names = ['bulge_smass', 'bulge_reff', 'disk_smass', 'disk_size']
    #names = ['disk_smass', 'disk_size']

    for i,name in enumerate(names):

        old = np.load(f"violin_data/{name}_old.npy")
        new =  np.load(f"violin_data/{name}.npy")

        old_boxes = np.load(f"violin_data/weights/{name}_old.npy")
        new_boxes = np.load(f"violin_data/weights/{name}.npy")

        old_cut = np.in1d(old_boxes, new_boxes)
        new_cut = np.in1d(new_boxes, old_boxes[old_cut])

        old_idx = np.argsort(old_boxes[old_cut])
        new_idx = np.argsort(new_boxes[new_cut])

        fig, ax = DREAMS.set_plot_params()

        ax.plot([-1000,1000], [1,1], 'r--', alpha=0.4)

        #ax.plot(old_boxes[old_cut][old_idx], old_boxes[old_cut][old_idx] - new_boxes[new_cut][new_idx], 'k.')
        #ax.plot(old[old_cut][old_idx], new[new_cut][new_idx], 'k.')
        ax.plot(old[old_cut][old_idx], old[old_cut][old_idx]/new[new_cut][new_idx], 'k.')
        ax.set_xlabel("Old")
        ax.set_ylabel("Old / New")

        old_data = np.clip(old[old_cut][old_idx], 0, 12)
        new_data = np.clip(new[new_cut][new_idx], 0, 12)
        print(name,np.mean(old_data), np.std(old_data), np.mean(new_data), np.std(new_data))

        ax.set_xlim((np.min(old[old_cut][old_idx]),np.max(old[old_cut][old_idx])))
        if i == 1:
        #    ax.set_xlim((0,2))
        #    ax.set_ylim((0,2))
            ax.set_ylim((0.2,5))
            ax.set_yscale("log")
            ax.set_xlim((0,1))
        if i == 0:
        #    ax.set_xlim((8.5,11))
        #    ax.set_ylim((8.5,11))           
            ax.set_ylim((0.8,1.2))
        if i == 2:
            ax.set_ylim((0.9,1.1))
        if i == 3:
            ax.set_ylim((0.2,5))
            ax.set_xlim((1,15))
            ax.set_yscale("log")
            ax.set_xscale("log")

        ax.set_title(name)

        fig.savefig(f"plots/{name}.pdf", bbox_inches='tight')


    return

if __name__=='__main__':
    main()
