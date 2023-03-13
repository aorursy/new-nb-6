import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import xgboost as xgb

from xgboost import XGBClassifier, XGBRegressor

from xgboost import plot_importance

from catboost import CatBoostRegressor,CatBoostClassifier

from matplotlib import pyplot

import shap

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

from time import time

from tqdm import tqdm

from collections import Counter

from scipy import stats

import lightgbm as lgb

from sklearn.metrics import cohen_kappa_score, mean_squared_error

from sklearn.model_selection import KFold, StratifiedKFold

import gc

import json

pd.set_option('display.max_columns', 1000)
def eval_qwk_lgb_regr(y_true, y_pred):

    """

    Fast cappa eval function for lgb.

    """

    dist = Counter(reduce_train['accuracy_group'])

    for k in dist:

        dist[k] /= len(reduce_train)

    reduce_train['accuracy_group'].hist()

    

    acum = 0

    bound = {}

    for i in range(3):

        acum += dist[i]

        bound[i] = np.percentile(y_pred, acum * 100)



    def classify(x):

        if x <= bound[0]:

            return 0

        elif x <= bound[1]:

            return 1

        elif x <= bound[2]:

            return 2

        else:

            return 3



    y_pred = np.array(list(map(classify, y_pred))).reshape(y_true.shape)



    return 'cappa', cohen_kappa_score(y_true, y_pred, weights='quadratic'), True
def cohenkappa(ypred, y):

    y = y.get_label().astype("int")

    ypred = ypred.reshape((4, -1)).argmax(axis = 0)

    loss = cohenkappascore(y, y_pred, weights = 'quadratic')

    return "cappa", loss, True
def read_data():

    print('Reading train.csv file....')

    #train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')

    print('Training.csv file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))



    print('Reading test.csv file....')

    test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')

    print('Test.csv file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))



    print('Reading train_labels.csv file....')

    train_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')

    print('Train_labels.csv file have {} rows and {} columns'.format(train_labels.shape[0], train_labels.shape[1]))



    print('Reading specs.csv file....')

    specs = pd.read_csv('/kaggle/input/data-science-bowl-2019/specs.csv')

    print('Specs.csv file have {} rows and {} columns'.format(specs.shape[0], specs.shape[1]))



    print('Reading sample_submission.csv file....')

    sample_submission = pd.read_csv('/kaggle/input/data-science-bowl-2019/sample_submission.csv')

    print('Sample_submission.csv file have {} rows and {} columns'.format(sample_submission.shape[0], sample_submission.shape[1]))

    return train, test, train_labels, specs, sample_submission
def encode_title(train, test, train_labels):

    # encode title

    train['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), train['title'], train['event_code']))

    test['title_event_code'] = list(map(lambda x, y: str(x) + '_' + str(y), test['title'], test['event_code']))

    all_title_event_code = list(set(train["title_event_code"].unique()).union(test["title_event_code"].unique()))

    # make a list with all the unique 'titles' from the train and test set

    list_of_user_activities = list(set(train['title'].unique()).union(set(test['title'].unique())))

    # make a list with all the unique 'event_code' from the train and test set

    list_of_event_code = list(set(train['event_code'].unique()).union(set(test['event_code'].unique())))

    list_of_event_id = list(set(train['event_id'].unique()).union(set(test['event_id'].unique())))

    # make a list with all the unique worlds from the train and test set

    list_of_worlds = list(set(train['world'].unique()).union(set(test['world'].unique())))

    # create a dictionary numerating the titles

    activities_map = dict(zip(list_of_user_activities, np.arange(len(list_of_user_activities))))

    activities_labels = dict(zip(np.arange(len(list_of_user_activities)), list_of_user_activities))

    activities_world = dict(zip(list_of_worlds, np.arange(len(list_of_worlds))))

    assess_titles = list(set(train[train['type'] == 'Assessment']['title'].value_counts().index).union(set(test[test['type'] == 'Assessment']['title'].value_counts().index)))

    # replace the text titles with the number titles from the dict

    train['title'] = train['title'].map(activities_map)

    test['title'] = test['title'].map(activities_map)

    train['world'] = train['world'].map(activities_world)

    test['world'] = test['world'].map(activities_world)

    train_labels['title'] = train_labels['title'].map(activities_map)

    win_code = dict(zip(activities_map.values(), (4100*np.ones(len(activities_map))).astype('int')))

    # then, it set one element, the 'Bird Measurer (Assessment)' as 4110, 10 more than the rest

    win_code[activities_map['Bird Measurer (Assessment)']] = 4110

    # convert text into datetime

    train['timestamp'] = pd.to_datetime(train['timestamp'])

    test['timestamp'] = pd.to_datetime(test['timestamp'])

    

    

    return train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code
# this is the function that convert the raw data into processed features

def get_data(user_sample, test_set=False):

    '''

    The user_sample is a DataFrame from train or test where the only one 

    installation_id is filtered

    And the test_set parameter is related with the labels processing, that is only requered

    if test_set=False

    '''

    # Constants and parameters declaration

    last_activity = 0

    

    user_activities_count = {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

    

    # new features: time spent in each activity

    last_session_time_sec = 0

    accuracy_groups = {0:0, 1:0, 2:0, 3:0}

    all_assessments = []

    accumulated_accuracy_group = 0

    accumulated_accuracy = 0

    accumulated_correct_attempts = 0 

    accumulated_uncorrect_attempts = 0

    accumulated_actions = 0

    counter = 0

    time_first_activity = float(user_sample['timestamp'].values[0])

    durations = []

    last_accuracy_title = {'acc_' + title: -1 for title in assess_titles}

    event_code_count: Dict[str, int] = {ev: 0 for ev in list_of_event_code}

    event_id_count: Dict[str, int] = {eve: 0 for eve in list_of_event_id}

    title_count: Dict[str, int] = {eve: 0 for eve in activities_labels.values()} 

    title_event_code_count: Dict[str, int] = {t_eve: 0 for t_eve in all_title_event_code}

        

    # last features

    sessions_count = 0

    

    # itarates through each session of one instalation_id

    for i, session in user_sample.groupby('game_session', sort=False):

        # i = game_session_id

        # session is a DataFrame that contain only one game_session

        

        # get some sessions information

        session_type = session['type'].iloc[0]

        session_title = session['title'].iloc[0]

        session_title_text = activities_labels[session_title]

                    

            

        # for each assessment, and only this kind off session, the features below are processed

        # and a register are generated

        if (session_type == 'Assessment') & (test_set or len(session)>1):

            # search for event_code 4100, that represents the assessments trial

            all_attempts = session.query(f'event_code == {win_code[session_title]}')

            # then, check the numbers of wins and the number of losses

            true_attempts = all_attempts['event_data'].str.contains('true').sum()

            false_attempts = all_attempts['event_data'].str.contains('false').sum()

            # copy a dict to use as feature template, it's initialized with some itens: 

            # {'Clip':0, 'Activity': 0, 'Assessment': 0, 'Game':0}

            features = user_activities_count.copy()

            features.update(last_accuracy_title.copy())

            features.update(event_code_count.copy())

            features.update(event_id_count.copy())

            features.update(title_count.copy())

            features.update(title_event_code_count.copy())

            features.update(last_accuracy_title.copy())

            features['installation_session_count'] = sessions_count

            

            variety_features = [('var_event_code', event_code_count),

                              ('var_event_id', event_id_count),

                               ('var_title', title_count),

                               ('var_title_event_code', title_event_code_count)]

            

            for name, dict_counts in variety_features:

                arr = np.array(list(dict_counts.values()))

                features[name] = np.count_nonzero(arr)

                 

            # get installation_id for aggregated features

            features['installation_id'] = session['installation_id'].iloc[-1]

            # add title as feature, remembering that title represents the name of the game

            features['session_title'] = session['title'].iloc[0]

            # the 4 lines below add the feature of the history of the trials of this player

            # this is based on the all time attempts so far, at the moment of this assessment

            features['accumulated_correct_attempts'] = accumulated_correct_attempts

            features['accumulated_uncorrect_attempts'] = accumulated_uncorrect_attempts

            accumulated_correct_attempts += true_attempts 

            accumulated_uncorrect_attempts += false_attempts

            # the time spent in the app so far

            if durations == []:

                features['duration_mean'] = 0

                features['duration_std'] = 0

            else:

                features['duration_mean'] = np.mean(durations)

                features['duration_std'] = np.std(durations)

            durations.append((session.iloc[-1, 2] - session.iloc[0, 2] ).seconds)

            # the accurace is the all time wins divided by the all time attempts

            features['accumulated_accuracy'] = accumulated_accuracy/counter if counter > 0 else 0

            accuracy = true_attempts/(true_attempts+false_attempts) if (true_attempts+false_attempts) != 0 else 0

            accumulated_accuracy += accuracy

            last_accuracy_title['acc_' + session_title_text] = accuracy

            # a feature of the current accuracy categorized

            # it is a counter of how many times this player was in each accuracy group

            if accuracy == 0:

                features['accuracy_group'] = 0

            elif accuracy == 1:

                features['accuracy_group'] = 3

            elif accuracy == 0.5:

                features['accuracy_group'] = 2

            else:

                features['accuracy_group'] = 1

            features.update(accuracy_groups)

            accuracy_groups[features['accuracy_group']] += 1

            # mean of the all accuracy groups of this player

            features['accumulated_accuracy_group'] = accumulated_accuracy_group/counter if counter > 0 else 0

            accumulated_accuracy_group += features['accuracy_group']

            # how many actions the player has done so far, it is initialized as 0 and updated some lines below

            features['accumulated_actions'] = accumulated_actions

            

            # there are some conditions to allow this features to be inserted in the datasets

            # if it's a test set, all sessions belong to the final dataset

            # it it's a train, needs to be passed throught this clausule: session.query(f'event_code == {win_code[session_title]}')

            # that means, must exist an event_code 4100 or 4110

            if test_set:

                all_assessments.append(features)

            elif true_attempts+false_attempts > 0:

                all_assessments.append(features)

                

            counter += 1

        

        sessions_count += 1

        # this piece counts how many actions was made in each event_code so far

        def update_counters(counter: dict, col: str):

                num_of_session_count = Counter(session[col])

                for k in num_of_session_count.keys():

                    x = k

                    if col == 'title':

                        x = activities_labels[k]

                    counter[x] += num_of_session_count[k]

                return counter

            

        event_code_count = update_counters(event_code_count, "event_code")

        event_id_count = update_counters(event_id_count, "event_id")

        title_count = update_counters(title_count, 'title')

        title_event_code_count = update_counters(title_event_code_count, 'title_event_code')



        # counts how many actions the player has done so far, used in the feature of the same name

        accumulated_actions += len(session)

        if last_activity != session_type:

            user_activities_count[session_type] += 1

            last_activitiy = session_type 

                        

    # if it't the test_set, only the last assessment must be predicted, the previous are scraped

    if test_set:

        return all_assessments[-1]

    # in the train_set, all assessments goes to the dataset

    return all_assessments
def get_train_and_test(train, test):

    compiled_train = []

    compiled_test = []

    for i, (ins_id, user_sample) in tqdm(enumerate(train.groupby('installation_id', sort = False)), total = 17000):

        compiled_train += get_data(user_sample)

    for ins_id, user_sample in tqdm(test.groupby('installation_id', sort = False), total = 1000):

        test_data = get_data(user_sample, test_set = True)

        compiled_test.append(test_data)

    reduce_train = pd.DataFrame(compiled_train)

    reduce_test = pd.DataFrame(compiled_test)

    categoricals = ['session_title']

    return reduce_train, reduce_test, categoricals
# read data

train, test, train_labels, specs, sample_submission = read_data()
# get usefull dict with maping encode

train, test, train_labels, win_code, list_of_user_activities, list_of_event_code, activities_labels, assess_titles, list_of_event_id, all_title_event_code = encode_title(train, test, train_labels)
# tranform function to get the train and test set

reduce_train, reduce_test, categoricals = get_train_and_test(train, test)
# call feature engineering function

features = reduce_train.loc[(reduce_train.sum(axis=1) != 0), (reduce_train.sum(axis=0) != 0)].columns # delete useless columns

features = [x for x in features if x not in ['accuracy_group', 'installation_id']]
counter = 0

to_remove = []

for feat_a in features:

    for feat_b in features:

        if feat_a != feat_b and feat_a not in to_remove and feat_b not in to_remove:

            c = np.corrcoef(reduce_train[feat_a], reduce_train[feat_b])[0][1]

            if c > 0.995:

                counter += 1

                to_remove.append(feat_b)

                print('{}: FEAT_A: {} FEAT_B: {} - Correlation: {}'.format(counter, feat_a, feat_b, c))
to_exclude = [] 

ajusted_test = reduce_test.copy()

for feature in ajusted_test.columns:

    if feature not in ['accuracy_group', 'installation_id', 'accuracy_group', 'session_title']:

        data = reduce_train[feature]

        train_mean = data.mean()

        data = ajusted_test[feature] 

        test_mean = data.mean()

        try:

            error = stract_hists(feature, adjust=True)

            ajust_factor = train_mean / test_mean

            if ajust_factor > 10 or ajust_factor < 0.1:# or error > 0.01:

                to_exclude.append(feature)

                print(feature, train_mean, test_mean, error)

            else:

                ajusted_test[feature] *= ajust_factor

        except:

            to_exclude.append(feature)

            print(feature, train_mean, test_mean)
features = [x for x in features if x not in (to_exclude + to_remove)]

reduce_train[features].shape
#Cat Model creation

from catboost import CatBoostRegressor,CatBoostClassifier

cat_clf=CatBoostClassifier(loss_function= 'MultiClass',task_type= "CPU",iterations= 200,

                   od_type= "Iter",depth= 10,colsample_bylevel= 0.5,early_stopping_rounds= 100,

                   l2_leaf_reg= 18,random_seed= 42,use_best_model= True)



cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

X=reduce_train[to_exclude]

Y=reduce_train['accuracy_group']

for train_idx, val_idx in cv.split(X,Y):

    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 

    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     

    cat_clf.fit(x_train,y_train,eval_set=(x_test, y_test),verbose=100)
#save models

import pickle

#cat_model

pickle.dump(cat_clf, open('cat.sav', 'wb'))
#xg model

xgb_clf=xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=1, gamma=0,

              learning_rate=0.05, max_delta_step=0, max_depth=10,

              min_child_weight=6, missing=None, n_estimators=200, n_jobs=1,

              nthread=None, objective='multi:softprob', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

X=reduce_train[to_exclude]

Y=reduce_train['accuracy_group']

for train_idx, val_idx in cv.split(X,Y):

    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 

    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     

    xgb_clf.fit(x_train,y_train)
#randomforest

from sklearn.ensemble import RandomForestClassifier

rf_clf=RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

                       max_depth=10, max_features='auto', max_leaf_nodes=None,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=4, min_samples_split=5,

                       min_weight_fraction_leaf=0.0, n_estimators=200,

                       n_jobs=None, oob_score=False, random_state=None,

                       verbose=0, warm_start=False)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

X=reduce_train[to_exclude]

Y=reduce_train['accuracy_group']

for train_idx, val_idx in cv.split(X,Y):

    x_train, y_train = X.iloc[train_idx], Y.iloc[train_idx] 

    x_test, y_test = X.iloc[val_idx], Y.iloc[val_idx]     

    rf_clf.fit(x_train,y_train)
#nn model

#cnn model
#meta data prepration

X=reduce_train[to_exclude]

X1=reduce_test[to_exclude]

Y=reduce_train['accuracy_group']

Y=Y.to_numpy()



#train data

tr1=cat_clf.predict_proba(X)

tr2=xgb_clf.predict_proba(X)

tr3=rf_clf.predict_proba(X)



#test data

ts1=cat_clf.predict_proba(X1)

ts2=xgb_clf.predict_proba(X1)

ts3=rf_clf.predict_proba(X1)



#merge the data

meta_data=np.column_stack((tr1,tr2,tr3))

meta_test_data=np.column_stack((ts1,ts2,ts3))
import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

tf.keras.backend.clear_session() ## For easy reset of notebook state

inputs = keras.Input(shape=(12,), name='ip')

x = layers.Dense(36, activation='sigmoid', name='dense_1')(inputs)

x = layers.Dense(24, activation='tanh', name='dense_4')(x)

x = layers.Dense(12, activation='relu', name='dense_7')(x)

x = layers.Dropout(.25)(inputs)

x = layers.Dense(8, activation='relu', name='dense_9')(x)

outputs = layers.Dense(4, activation='softmax', name='predictions')(x)

meta_clf = keras.Model(inputs=inputs, outputs=outputs)

meta_clf.compile(optimizer=keras.optimizers.RMSprop(),  # Optimizer

              loss=keras.losses.SparseCategoricalCrossentropy(), # Loss function to minimize

              metrics=[keras.metrics.SparseCategoricalAccuracy()]) # List of metrics to monitor
#data split up

from sklearn.model_selection import train_test_split

x1_train, x1_test, y1_train, y1_test = train_test_split(meta_data,Y, test_size=0.3)

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

for train_idx, val_idx in cv.split(x1_train,y1_train):

    x_train, y_train = x1_train[train_idx], y1_train[train_idx] 

    x_test, y_test = x1_train[val_idx], y1_train[val_idx] 

    history = meta_clf.fit(x_train, y_train,validation_data=(x_test, y_test),batch_size=24,epochs=50)

    #print('\nhistory dict:', history.history)

    
print('\nhistory dict:', history.history)
results = meta_clf.evaluate(x_test, y_test, batch_size=32)

pred=meta_clf.predict(x_test)

f_pred=[]

for i in pred:

    indices = np.where(i == i.max())

    f_pred.append(int(indices[0]))
from sklearn.metrics import cohen_kappa_score

kappa = cohen_kappa_score(f_pred,y_test,weights='quadratic')

print('test loss, test acc:', results)

print('Kappa Score:', kappa)
#predict for the test data

pred1=meta_clf.predict(meta_test_data)

final_pred=[]

for i in pred1:

    indices = np.where(i == i.max())

    final_pred.append(int(indices[0]))
sample_submission['accuracy_group'] = final_pred

sample_submission.to_csv('submission.csv', index=False)
sample_submission.groupby(['accuracy_group']).count()