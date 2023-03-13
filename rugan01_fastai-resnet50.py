# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import shutil

import cv2

import re

from fastai import *

from fastai.vision import *

from fastai.metrics import error_rate


bs = 64
path = Path('/kaggle/input/Kannada-MNIST')

train = pd.read_csv('/kaggle/input/Kannada-MNIST/train.csv')

test  =pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

sub = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

dig = pd.read_csv('/kaggle/input/Kannada-MNIST/Dig-MNIST.csv')
outpath = '/kaggle/'

os.listdir(outpath)
#creating the output folder for storing train and test images



shutil.os.mkdir(outpath + "output")

shutil.os.mkdir(outpath + "output/train")

shutil.os.mkdir(outpath + "output/test")

shutil.os.mkdir(outpath + "output/model")

print(train.shape); print(test.shape); print(dig.shape)
print(train.isna().sum().sum());print(test.isna().sum().sum());print(dig.isna().sum().sum())
#train.groupby('label').describe()
#Function to save the images to re-use in fastai

def save_images(outpath,df):

    for index, row in df.iterrows():

        pixels=np.asarray(row['pixel0':'pixel783'])

        img=pixels.reshape((28,28))

        pathname=os.path.join(outpath,'image_'+str(row['label'])+'_'+str(index)+'.jpg')

        cv2.imwrite(pathname,img)
outpath = '/kaggle/output/train'

save_images(outpath,train)
#sample_image = open_image('/kaggle/output/train/image_5_39645.jpg')

#sample_image.show(figsize=(4,4))
path_img =  Path(outpath)

fnames = get_image_files(path_img)

fnames[:5]
pat = r'/image_(\d+)_'

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=4).normalize(imagenet_stats)
data.show_batch(rows=5, figsize=(15,8))
print(data.classes)

len(data.classes),data.c

#!cp /kaggle/input/resnet34/resnet34.pth /tmp/.cache/torch/checkpoints/resnet34-333f7ec4.pth

learn = cnn_learner(data, models.resnet50, metrics=[error_rate, accuracy], model_dir = '/kaggle/output/model')
learn.fit_one_cycle(1)
#learn.save('stage1')
#interp = ClassificationInterpretation.from_learner(learn)

#interp.most_confused(min_val=2)
learn.unfreeze()
#learn.lr_find()

#learn.recorder.plot()
#learn.fit_one_cycle(2, max_lr=slice(3e-6,3e-4))
#interp = ClassificationInterpretation.from_learner(learn)

#interp.most_confused(min_val=2)
#learn.save('stage2')
learn.fit_one_cycle(14,3e-6)

learn.save('stage3')
interp = ClassificationInterpretation.from_learner(learn)

interp.most_confused(min_val=2)
submit = pd.DataFrame(columns=['id','label'])

submit['id'] = test['id']
for index, row in test.iterrows():

    pixels=np.asarray(row['pixel0':'pixel783'])

    arr=pixels.reshape((28,28))/255

    arr = np.stack([arr]*3,axis=0)

    img = Image(FloatTensor(arr))

    submit.at[index, 'label'] = int(learn.predict(img)[1])
submit.head(20)
submit.to_csv('submission.csv',index=False)