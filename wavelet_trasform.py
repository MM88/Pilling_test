import sys
import numpy as np
import dtcwt
import cv2
import plot_utils
from sklearn import datasets
import matplotlib.pyplot as plt



def energy_vector(hp):
    e_vec = []
    for i in range(hp.shape[2]):
        e_vec.append(np.angle(hp[:,:,i]).std())
    return e_vec

def main(argv):


    # image2 = cv2.imread('pill2.jpg')
    # gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("./pill2_gray.jpg",gray_image)

    image = plt.imread('pill1_gray.jpg')
    plt.figure("original image")
    plt.imshow(image, cmap = plt.cm.gray)

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_image',gray_image)
    # cv2.waitKey(0)
    transform = dtcwt.Transform2d()
    pill_t = transform.forward(image, nlevels=8 ,include_scale=True )

    # # draw the approximation at a wanted level
    plot_utils.plot_approximation_image(pill_t, level = 2)

    # # draw the reconstruction at a wanted level, level 0 is the original image
    plot_utils.plot_reconstructed_detail(pill_t, level = 2)

    # # draw each orientation at each trasform level and the resulting image at each level
    # plot_utils.plot_trasform_levels(pill_t, len(pill_t.highpasses))

    # #compute energy vector for levels 5 and 6
    # e_vec_mat = []
    # for i in range(ndataset):
    #      e_vec = energy_vector(pill_t.highpasses[4]) + energy_vector(pill_t.highpasses[5])
    #      e_vec_mat = np.concatenate((e_vec_mat, e_vec), axis=0)
    #
    # # e_vec_mat = np.concatenate((energy_vector(pill_t.highpasses[4]), energy_vector(pill_t.highpasses[5])), axis=0) #crea matrice con due righe
    #
    # #perform PCA on energy vector, PCA vuole come input una matrice in cui ogni riga e' un feature vector
    # pca = decomposition.PCA(n_components=2)
    # pca.fit(e_vec_mat)
    # e_vec_mat = pca.transform(e_vec_mat)

    plt.show()
    pass

if __name__ == "__main__":
    main(sys.argv)