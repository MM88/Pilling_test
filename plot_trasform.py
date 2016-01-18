import numpy as np
import matplotlib.pyplot as plt

def plot_trasform_levels(image_wt, nlevels):

    # Show the absolute images for each direction.
    # Note that the 2nd level has index 1 since the 1st has index 0.
    for i in range(nlevels):
        plt.figure(i)
        for slice_idx in range(image_wt.highpasses[i].shape[2]):
            plt.subplot(2, 3, slice_idx)
            plt.imshow(np.angle(image_wt.highpasses[i][:,:,slice_idx]), cmap=plt.cm.gray, clim=(-np.pi, np.pi))

     # reconstruction from filters as sum of values in each direction
    for i in range(nlevels):
        plt.figure(i+nlevels)
        rec_0 = np.angle(image_wt.highpasses[i][:,:,0])
        for slice_idx in range(image_wt.highpasses[i].shape[2]-1):
            rec_0 = rec_0 + np.angle(image_wt.highpasses[i][:,:,slice_idx+1])
        plt.imshow(rec_0, cmap=plt.cm.gray, clim=(-np.pi, np.pi))

    plt.show()

    return