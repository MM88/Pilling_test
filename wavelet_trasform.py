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

def compute_and_plot_transform(image):

  # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X, Y, Z = axes3d.get_test_data(0.05)
    # ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

    # image2 = cv2.imread('./PILLING/IMG_0822_GS.JPG')
    # gray_image = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # # grey = image2[0:image2.shape[0], 0:image2.shape[0]]
    # # cv2.imshow("cc",grey)
    # # cv2.waitKey(0)
    # cv2.imwrite("./PILLING/IMG_0822_GS.JPG",gray_image)


    # print np.cov(image)
    # fig = plt.figure( "image")
    # x,y = np.mgrid[:image.shape[0],:image.shape[1]]
    # ax2 = fig.add_subplot(1,1,1,projection='3d')
    # ax2.plot_surface(x,y,image,cmap=plt.cm.jet,rstride=1,cstride=1,linewidth=0.,antialiased=False)
    # ax2.set_title('3D')
    # ax2.set_zlim3d(0,1000)

    plt.figure("original image")
    plt.imshow(image, cmap = plt.cm.gray)

    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray_image',gray_image)
    # cv2.waitKey(0)
    transform = dtcwt.Transform2d()
    pill_t = transform.forward(image, nlevels = 6 ,include_scale=True )

    # # draw the approximation at a wanted level
    # plot_utils.plot_approximation_image(pill_t, level = 0)
    res_img = plot_utils.plot_approximation_image(pill_t, level = 4)

    # # draw the reconstruction at a wanted level, level 0 is the original image
    # plot_utils.plot_reconstructed_detail(pill_t, level = 0)
    plot_utils.plot_reconstructed_detail(pill_t, level = 4)

    # # draw each orientation at each trasform level and the resulting image at each level
    # plot_utils.plot_trasform_levels(pill_t, len(pill_t.highpasses))

    # #compute energy vector for levels 5 and 6
    # e_vec_mat = []
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



    return res_img

def main(argv):

# 21/21P, 33, 32, 31, 22

    # image1 = plt.imread('./dataset/1/pill_14.jpg')
    # res_img1 = compute_and_plot_transform(image1)
    # image2 = plt.imread('./dataset/3/pill_34.jpg')
    # res_img2 = compute_and_plot_transform(image2)
    # image3 = plt.imread('./dataset/5/pill_14.jpg')
    # res_img3 = compute_and_plot_transform(image3)
    #
    # print np.array(res_img1).max(), np.array(res_img2).max(), np.array(res_img3).max()
    # print np.array(res_img1).mean(),np.array(res_img1).std()
    # print np.array(res_img2).mean(), np.array(res_img2).std()
    # print np.array(res_img3).mean(), np.array(res_img3).std()


    from skimage import exposure
    import skimage.morphology as morp
    from skimage.filters import rank


    img = plt.imread('./dataset/1/pill_14.jpg')

    # Global equalize
    img_global = exposure.equalize_hist(img)
    res_img1 = compute_and_plot_transform(img_global)

    image2 = plt.imread('./dataset/3/pill_34.jpg')

    img_global2 = exposure.equalize_hist(image2)
    res_img2 = compute_and_plot_transform(img_global2)

    print np.array(res_img1).max(), np.array(res_img2).max()
    print np.array(res_img1).mean(),np.array(res_img1).std()
    print np.array(res_img2).mean(), np.array(res_img2).std()


    # # Local Equalization, disk shape kernel
    # # Better contrast with disk kernel but could be different
    # kernel = morp.disk(30)
    # img_local = rank.equalize(img, selem=kernel)
    #
    # fig, (ax_img, ax_global, ax_local) = plt.subplots(1, 3)
    #
    # ax_img.imshow(img, cmap=plt.cm.gray)
    # ax_img.set_title('Low contrast image')
    # ax_img.set_axis_off()
    #
    # ax_global.imshow(img_global, cmap=plt.cm.gray)
    # ax_global.set_title('Global equalization')
    # ax_global.set_axis_off()
    #
    # ax_local.imshow(img_local, cmap=plt.cm.gray)
    # ax_local.set_title('Local equalization')
    # ax_local.set_axis_off()


    plt.show()

    pass

if __name__ == "__main__":
    main(sys.argv)