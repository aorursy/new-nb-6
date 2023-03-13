# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from collections import defaultdict



usecols = ['ncodpers','fecha_dato', 'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',

       'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',

       'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',

       'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',

       'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',

       'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',

       'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',

       'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']

       

df_train = pd.read_csv('../input/train_ver2.csv', usecols=usecols)

df_test = pd.read_csv('../input/test_ver2.csv', usecols= ['ncodpers','fecha_dato'])
unique_train = df_train['ncodpers'].unique()
unique_test = df_test['ncodpers'].unique()
df_test.ncodpers.isin(unique_train).sum()
df_test.shape
unique_train
df_train[df_train.ncodpers == 1170544]
unique_test
dftr_prev =  df_train[df_train['fecha_dato'].isin(['2016-04-28'])]
dftr_curr = df_train[df_train['fecha_dato'].isin(['2016-05-28'])]
dftr_prev.drop(['fecha_dato'], axis=1, inplace = True)

dftr_curr.drop(['fecha_dato'], axis=1, inplace = True)



dfm = pd.merge(dftr_curr,dftr_prev, how='inner', on=['ncodpers'], suffixes=('', '_prev'))
dfm.columns
dfm.shape