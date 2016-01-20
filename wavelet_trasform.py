import sys
import numpy as np
import dtcwt
import cv2
import plot_utils
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import axes3d



def energy_vector(hp):
    e_vec = []
    for i in range(hp.shape[2]):
        e_vec.append(np.angle(hp[:,:,i]).std())
    return e_vec

def main(argv):



    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.05)
    # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # image2 = cv2.imread('bluepill_gray_sqr.jpg')
    # gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # grey = image2[0:image2.shape[0], 0:image2.shape[0]]
    # cv2.imshow("cc",grey)
    # cv2.waitKey(0)
    # cv2.imwrite("./bluepill_gray_sqr.jpg",gray_image)


    image = plt.imread('bluepill_gray_sqr.jpg')

    # x,y = np.mgrid[:image.shape[0],:image.shape[1]]
    # ax2 = fig.add_subplot(111,projection='3d')
    # ax2.plot_surface(x,y,image,cmap=plt.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    # ax2.set_title('3D')
    # ax2.set_zlim3d(0,1000)


    plt.figure("original image")
    plt.imshow(image, cmap = plt.cm.gray)

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_image',gray_image)
    # cv2.waitKey(0)
    transform = dtcwt.Transform2d()
    pill_t = transform.forward(image, nlevels=8 ,include_scale=True )

    # # draw the approximation at a wanted level
    plot_utils.plot_approximation_image(pill_t, level = 1)

    # # draw the reconstruction at a wanted level, level 0 is the original image
    plot_utils.plot_reconstructed_detail(pill_t, level = 1)

    # # draw each orientation at each trasform level and the resulting image at each level
    # plot_utils.plot_trasform_levels(pill_t, len(pill_t.highpasses))

    # #compute energy vector for levels 5 and 6
    e_vec_mat = []
    # for i in range(1):
    #      e_vec = energy_vector(pill_t.highpasses[4]) + energy_vector(pill_t.highpasses[5])
    #      e_vec2 = energy_vector(pill_t.highpasses[2]) + energy_vector(pill_t.highpasses[3])
    #      e_vec3 = energy_vector(pill_t.highpasses[0]) + energy_vector(pill_t.highpasses[1])
    #      e_vec4 = energy_vector(pill_t.highpasses[6]) + energy_vector(pill_t.highpasses[7])
    #      # e_vec_mat = np.concatenate((e_vec_mat, e_vec), axis=0)
    #
    # print e_vec
    # print e_vec2
    # print e_vec3
    # print e_vec4

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