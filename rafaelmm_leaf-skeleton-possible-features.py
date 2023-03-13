# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import scipy.ndimage as ndi

from skimage.morphology import skeletonize

from skimage import measure

from keras.preprocessing.image import load_img, img_to_array



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# ------------------ Init ----------------------------------

#img = mpimg.imread('../input/images/53.jpg')

img = load_img('../input/images/317.jpg', grayscale=True)
max_ax = max((0, 1), key=lambda x: img.size[x])

scale = 500 / float(img.size[max_ax])

img = img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))

img = img_to_array(img)

img = np.around(img / 255.0)

print(img.shape)

# --------------------------------- Feature Extraction -------

#img = np.divide(img,np.max(np.max(img)))

img = img.squeeze()

skeleton = skeletonize(img)

print(type(skeleton))

print(img.shape)

print(skeleton.shape)



# --------------------------------- Display The Results -------

# display results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True, subplot_kw={'adjustable':'box-forced'})



ax1.imshow(img, cmap=plt.cm.gray)

ax1.axis('off')

ax1.set_title('original', fontsize=20)



ax2.imshow(skeleton, cmap=plt.cm.gray)

ax2.axis('off')

ax2.set_title('skeleton', fontsize=20)



fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.98,

                    bottom=0.02, left=0.02, right=0.98)



plt.show()