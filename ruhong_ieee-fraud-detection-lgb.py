# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import os

import random

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from lightgbm import LGBMClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV, PredefinedSplit

SEED = 31

N_ESTIMATORS = 20000

TARGET = 'isFraud'

VALIDATION_PERCENT = 0.16

SCORING = 'roc_auc'
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    

seed_everything(SEED)
file_folder = '../input/ieee-fraud-detection-preprocess'

train = pd.read_csv(f'{file_folder}/train.csv')

test = pd.read_csv(f'{file_folder}/test.csv')

print(f'train={train.shape}, test={test.shape}')
#excludes = {TARGET}

#for i in range(1, 340):

 #   excludes.add(f'V{i}')

#cols = set(train.columns.values) - excludes
def _keep(col):

    if col == TARGET:

        return False

    if col.startswith('_pc_'):

        return False

    if '_to_' in col:

        return False

    return True





PREDICTORS = [c for c in train.columns.values if _keep(c)]

print(f'{len(PREDICTORS)} predictors={PREDICTORS}')
val_size = int(VALIDATION_PERCENT * len(train))

train_size = len(train) - val_size

train_ind = [-1] * train_size

val_ind = [0] * val_size

ps = PredefinedSplit(test_fold=np.concatenate((train_ind, val_ind)))

val = train[-val_size:]

print(f'val={val.shape}')

y_train = train[TARGET]

x_train = train[PREDICTORS]

y_val = val[TARGET]

x_val = val[PREDICTORS]

model = LGBMClassifier(learning_rate=0.1, n_estimators=N_ESTIMATORS, reg_alpha=1)

pipe = Pipeline([('model', model)])

param_grid = {

    'model__num_leaves': [80],

    'model__min_child_samples': [1000],

    'model__colsample_bytree': [0.5]

}

cv = GridSearchCV(pipe, cv=ps, param_grid=param_grid, scoring=SCORING)

cv.fit(x_train, y_train, model__eval_set=[(x_val, y_val)], model__eval_metric='auc', model__early_stopping_rounds=200, model__verbose=500)

print('best_params_={}\nbest_score_={}'.format(repr(cv.best_params_), repr(cv.best_score_)))
x_test = test[PREDICTORS]

sub = pd.read_csv(f'../input/ieee-fraud-detection/sample_submission.csv')

sub[TARGET] = cv.predict_proba(x_test)[:,1]

sub.head()
sub.to_csv('submission.csv', index=False)

print(os.listdir("."))