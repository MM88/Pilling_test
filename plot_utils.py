import numpy as np
import matplotlib.pyplot as plt
import cv2
import dtcwt
from dtcwt.numpy import Pyramid


def plot_trasform_levels(image_wt, nlevels):

    # Show the absolute images for each direction.
    # Note that the 2nd level has index 1 since the 1st has index 0.
    for i in range(nlevels):
        plt.figure('angles'+str(i))
        for slice_idx in range(image_wt.highpasses[i].shape[2]):
            plt.subplot(2, 3, slice_idx)
            plt.imshow(np.angle(image_wt.highpasses[i][:,:,slice_idx]),cmap=plt.cm.gray, clim=(-np.pi, np.pi))

    # reconstruction from filters as sum of values in each direction
    for i in range(nlevels):
        plt.figure('angle sum' + str(i))
        rec_0 = np.angle(image_wt.highpasses[i][:,:,0])
        for slice_idx in range(image_wt.highpasses[i].shape[2]-1):
            rec_0 = rec_0 + np.angle(image_wt.highpasses[i][:,:,slice_idx+1])
        plt.imshow(rec_0, cmap=plt.cm.gray,  clim=(-np.pi, np.pi))

    return

def plot_approximation_image(image_wt, level):

    plt.figure( "approx at level " + str(level))
    plt.imshow(image_wt.scales[level], cmap=plt.cm.gray)

    # im = np.array(image_wt.scales[level])
    # window_name = "approx at level " + str(level)
    # cv2.imshow(window_name,im)
    # cv2.waitKey(0)

    mean = image_wt.scales[level].mean()
    std = image_wt.scales[level].std()
    abs_img = [[cell if (mean-cell) > 2*std else 0.0 for cell in row] for row in image_wt.scales[level] ]
    blue_img = [[(cell) if cell < mean else mean for cell in row] for row in image_wt.scales[level] ]
    blue_img = [[(mean-cell) if cell != mean else 0 for cell in row] for row in blue_img]
    # blue_img =  cv2.GaussianBlur(np.array(blue_img),(5,5),0) #smoothing gaussiano


    # abs_img = [(row[i]+mean) if (row[i] < mean) else row[i] for i in image.shape[] for row in image ]

    fig = plt.figure('3d level'+ str(level))
    x,y = np.mgrid[:image_wt.scales[level].shape[0],:image_wt.scales[level].shape[1]]
    ax2 = fig.add_subplot(1,1,1,projection='3d')
    ax2.plot_surface(x,y,blue_img,cmap=plt.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    # ax2.plot_surface(x,y, image_wt.scales[level],cmap=plt.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    ax2.set_title('3D approx '+str(level))
    ax2.set_zlim3d(0,1000)

    return blue_img

def plot_reconstructed_detail(image_wt, level):

    transform = dtcwt.Transform2d()
    yh = []
    yh.append(image_wt.highpasses[level])
    yh = tuple(yh)
    py = Pyramid(image_wt.scales[level], yh )
    inv_pill_t = transform.inverse(py)

    plt.figure( "recon at level " + str(level))
    plt.imshow(inv_pill_t, cmap=plt.cm.gray)

    plt.figure( "details plus approx at level " + str(level))
    plt.imshow(inv_pill_t + image_wt.scales[level], cmap=plt.cm.gray)

    plt.figure( "details minus approx at level " + str(level))
    plt.imshow( inv_pill_t - image_wt.scales[level], cmap=plt.cm.gray)

    #
    # orig = transform.inverse(image_wt)
    # sub = orig - (inv_pill_t - image_wt.scales[level])
    # plt.figure( " image minus details minus approx at level " + str(level))
    # plt.imshow(sub , cmap=plt.cm.gray)

    # fig = plt.figure()
    # x,y = np.mgrid[:sub.shape[0],:sub.shape[1]]
    # ax2 = fig.add_subplot(1,1,1,projection='3d')
    # ax2.plot_surface(x,y,sub,cmap=plt.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    # ax2.set_title('3D recon sub'+str(level))
    # ax2.set_zlim3d(0,1000)
    #
    # fig = plt.figure()
    # x,y = np.mgrid[:orig.shape[0],:orig.shape[1]]
    # ax2 = fig.add_subplot(1,1,1,projection='3d')
    # ax2.plot_surface(x,y,orig,cmap=plt.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    # ax2.set_title('3D recon orig'+str(level))
    # ax2.set_zlim3d(0,1000)


    return




