import numpy as np
import os
import matplotlib.pyplot as plt
import re
for img_file in os.listdir():
    if re.match(r"^0.*\.jpg$",img_file):
        img = plt.imread(img_file)
        M = int(img.shape[0]/3)
        imgB = img[0:M, ...]
        imgG = img[M:2*M, ...]
        imgR = img[2*M:3*M, ...]
        img_out = np.dstack((imgR, imgG, imgB))
        plt.imsave('rgb_'+img_file, img_out)
