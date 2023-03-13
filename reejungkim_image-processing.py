# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
pd.read_csv("../input/sample_submission.csv").head()
images = os.listdir("../input/test")

images[:10]
import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import cv2





#read the first jpg file

img = cv2.imread('../input/test/b4c3b52a8723d431.jpg',0)

#img = cv2.imread('../input/test/b4c3b52a8723d431.jpg')



#check the array of the first jpg file

img
#view the array as an image

plt.imshow(img)
plt.imshow(img, cmap=plt.cm.gray)
a=cv2.imread('../input/test/631bb0244ef594e6.jpg')

plt.imshow(a)

#a
plt.imshow( cv2.imread('../input/test/584541c62119b661.jpg') )

#plt.imshow( cv2.imread('../input/test/584541c62119b661.jpg', 0 ) ) 
x= '../input/test/'

myList = [ x + i for i in images[:10]]
myList
for i in myList:

    plt.imshow( cv2.imread(i) ) 

    plt.show()