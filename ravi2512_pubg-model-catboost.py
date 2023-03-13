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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
TrainingData = pd.read_csv('../input/train.csv')
TrainingData.columns
X=TrainingData.values[:,3:-1]
Y=TrainingData.values[:,-1]
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.01)
import catboost
regressor=catboost.CatBoostRegressor(iterations=300,learning_rate=0.9,eval_metric='MAE')
regressor.fit(X_train,y_train,eval_set=(X_test,y_test))
result=mean_absolute_error(y_test,regressor.predict(X_test))
result
test_data_org = pd.read_csv('../input/test.csv')
test_data=test_data_org.values[:,3:]
y_pred=regressor.predict(test_data)
submission=pd.DataFrame()
submission['Id']=test_data_org['Id']
submission['winPlacePerc']=y_pred
submission.to_csv('submission.csv',index=False)




