# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import seaborn as sns

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
sns.distplot(np.log1p(train['target']))
def to_mape(msle):
    return np.exp(msle) - 1

msles = np.asarray(range(1,36))/10
plt.plot(msles,to_mape(msles))
plt.xlabel("MSLE")
plt.ylabel("MAPE")
