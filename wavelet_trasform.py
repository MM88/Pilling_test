import numpy as np
import matplotlib.pyplot as plt
import dtcwt
from dtcwt import compat
from sklearn import decomposition
from sklearn import datasets
import cv2
import pywt
from dtcwt.numpy import Pyramid
import plot_trasform

def energy_vector(hp):
    e_vec = []
    for i in range(hp.shape[2]):
        e_vec.append(np.angle(hp[:,:,i]).std())
    return e_vec

import sys

def main(argv):

    image = cv2.imread('large.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray_image',gray_image)
    cv2.waitKey(0)
    transform = dtcwt.Transform2d()
    pill_t = transform.forward(gray_image, nlevels=8 ,include_scale=True )
    # perform and draw antitrasform
    pill_recon = transform.inverse(pill_t)
    im = np.array(pill_recon, dtype = np.uint8)
    cv2.imshow("reconstructed",im)
    cv2.waitKey(0)
    
    #draw each orientation at each trasform level and the resulting image at each level
    # plot_trasform.plot_trasform_levels(pill_t, len(pill_t.highpasses))

    pass

if __name__ == "__main__":
    main(sys.argv)