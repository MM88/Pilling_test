import cPickle
import matplotlib.pyplot as plt
import numpy as np
import wavelet_trasform
import glob
from skimage import exposure
from sklearn import preprocessing
# #
class Pilled:
    def __init__(self):
        self.x = []
        self.y = []
    pass
#
# img = Pilled()
#
# for folder in glob.glob('./dataset/*'): # per ogni cartella del fataset
#     for filename in glob.glob(folder+'/*.jpg'): # per ogni immagine della cartella
#         im= plt.imread(filename) # legge l'immagine
#         # img_global = exposure.equalize_hist(im)
#         img_wt = wavelet_trasform.compute_and_plot_transform(im) # ne calcola la trasformata al 4 livello
#         img_wt = preprocessing.normalize(img_wt)
#         fvector = []
#         fvector.append(np.array(img_wt).max())
#         fvector.append(np.array(img_wt).mean())
#         fvector.append(np.array(img_wt).std())
#         img.x.append(fvector) # appende trasformata e classe di appartenenza nel pkl
#         img.y.append(folder[-1:])
#
#
# #
# filehandler = open("./features_dataset.pkl","wb")
# cPickle.dump(img,filehandler)
# filehandler.close()
# #
file = open("./features_dataset.pkl",'rb')
object_file = cPickle.load(file)
file.close()


print(object_file.x, object_file.y)





