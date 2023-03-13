import pandas as pd

import numpy as np

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler, LabelBinarizer

import xgboost as xgb

class FeatureBinarizatorAndScaler:

    """ This class needed for scaling and binarization features

    """

    NUMERICAL_FEATURES = list()

    CATEGORICAL_FEATURES = list()

    BIN_FEATURES = list()

    binarizers = dict()

    scalers = dict()



    def __init__(self, numerical=list(), categorical=list(), binfeatures = list(), binarizers=dict(), scalers=dict()):

        self.NUMERICAL_FEATURES = numerical

        self.CATEGORICAL_FEATURES = categorical

        self.BIN_FEATURES = binfeatures

        self.binarizers = binarizers

        self.scalers = scalers



    def fit(self, train_set):

        for feature in train_set.columns:



            if feature.split('_')[-1] == 'cat':

                self.CATEGORICAL_FEATURES.append(feature)

            elif feature.split('_')[-1] != 'bin':

                self.NUMERICAL_FEATURES.append(feature)



            else:

                self.BIN_FEATURES.append(feature)

        for feature in self.NUMERICAL_FEATURES:

            scaler = StandardScaler()

            self.scalers[feature] = scaler.fit(np.float64(train_set[feature]).reshape((len(train_set[feature]), 1)))

        for feature in self.CATEGORICAL_FEATURES:

            binarizer = LabelBinarizer()

            self.binarizers[feature] = binarizer.fit(train_set[feature])





    def transform(self, data):

        binarizedAndScaledFeatures = np.empty((0, 0))

        for feature in self.NUMERICAL_FEATURES:

            if feature == self.NUMERICAL_FEATURES[0]:

                binarizedAndScaledFeatures = self.scalers[feature].transform(np.float64(data[feature]).reshape(

                    (len(data[feature]), 1)))

            else:

                binarizedAndScaledFeatures = np.concatenate((

                    binarizedAndScaledFeatures,

                    self.scalers[feature].transform(np.float64(data[feature]).reshape((len(data[feature]),

                                                                                       1)))), axis=1)

        for feature in self.CATEGORICAL_FEATURES:



            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures,

                                                         self.binarizers[feature].transform(data[feature])), axis=1)



        for feature in self.BIN_FEATURES:

            binarizedAndScaledFeatures = np.concatenate((binarizedAndScaledFeatures, np.array(data[feature]).reshape((

                len(data[feature]), 1))), axis=1)

        print(binarizedAndScaledFeatures.shape)

        return binarizedAndScaledFeatures
def gini(actual, pred, cmpcol=0, sortcol=1):

    assert (len(actual) == len(pred))

    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)

    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]

    totalLosses = all[:, 0].sum()

    giniSum = all[:, 0].cumsum().sum() / totalLosses



    giniSum -= (len(actual) + 1) / 2.

    return giniSum / len(actual)





def gini_normalized(a, p):

    return gini(a, p) / gini(a, a)





def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    gini_score = gini_normalized(labels, preds)

    return [('gini', gini_score)]

first = pd.read_csv('../input/results/5folds(v. 1).csv')

second  = pd.read_csv('../input/results/5folds(v. 2).csv')

third =  pd.read_csv('../input/results/4folds.csv')

fourth = pd.read_csv('../input/results/LightGBM.csv')

fifth = pd.read_csv('../input/results/NN.csv')



a = pd.DataFrame()

a['id'] = first['id']

a['target'] = (first['target']+second['target']+third['target']+fourth['target']+fifth['target'])/5

a.to_csv('submission.csv')
def preproc(X_train):

    # Adding new features and deleting features with low importance

    multreg = X_train['ps_reg_01'] * X_train['ps_reg_03'] * X_train['ps_reg_02']

    ps_car_reg = X_train['ps_car_13'] * X_train['ps_reg_03'] * X_train['ps_car_13']

    X_train = X_train.drop(['ps_calc_01', 'ps_calc_02', 'ps_calc_03', 'ps_calc_04', 'ps_calc_05', 'ps_calc_06',

                            'ps_calc_07', 'ps_calc_08', 'ps_calc_09', 'ps_calc_10', 'ps_calc_11', 'ps_calc_12',

                            'ps_calc_13', 'ps_calc_14', 'ps_calc_15_bin', 'ps_calc_16_bin', 'ps_calc_17_bin',

                            'ps_calc_18_bin', 'ps_calc_19_bin', 'ps_calc_20_bin', 'ps_car_10_cat', 'ps_ind_10_bin',

                            'ps_ind_13_bin', 'ps_ind_12_bin'], axis=1)

    X_train['mult'] = multreg

    X_train['ps_car'] = ps_car_reg

    X_train['ps_ind'] = X_train['ps_ind_03'] * X_train['ps_ind_15']

    return X_train

X_train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

y_train = X_train['target']

X_train = X_train.drop(['id', 'target'], axis=1)

X_test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

X_test = X_test.drop(['id'], axis=1)

X_train = preproc(X_train)

X_test = preproc(X_test)



binarizerandscaler = FeatureBinarizatorAndScaler()

binarizerandscaler.fit(X_train)

X_train = binarizerandscaler.transform(X_train)

X_test = binarizerandscaler.transform(X_test)



# Kinetic features https://www.kaggle.com/alexandrudaia/kinetic-and-transforms-0-482-up-the-board

kinetic_train = []

for i in range(4):

    kinetic_train = pd.read_csv('../input/kinetic-features/'+str(i+1)+'k.csv').iloc[:, 1:2]

    X_train = np.concatenate((X_train, np.array(kinetic_train).reshape((len(kinetic_train), 1))), axis=1)

    kinetic_test = pd.read_csv('../input/kinetic-features/'+str(i+1)+'kt.csv').iloc[:, 1:2]

    X_test = np.concatenate((X_test, np.array(kinetic_test).reshape((len(kinetic_test), 1))), axis=1)



i = 0

K = 5

kf = KFold(n_splits=K, random_state=42, shuffle=True)

# 5 Cross Validation

results = []

for train_index, test_index in kf.split(X_train):

    train_X, valid_X = X_train[train_index], X_train[test_index]

    train_y, valid_y = y_train[train_index], y_train[test_index]

    weights = np.zeros(len(y_train))

    weights[y_train == 0] = 1

    weights[y_train == 1] = 1

    print(weights, np.mean(weights))

    watchlist = [(xgb.DMatrix(train_X, train_y, weight=weights), 'train'), (xgb.DMatrix(valid_X, valid_y), 'valid')]

    # Setting parameters for XGBoost model

    params = {'eta': 0.03, 'max_depth': 4, 'objective': 'binary:logistic', 'seed': 42, 'silent': True}

    model = xgb.train(params, xgb.DMatrix(train_X, train_y, weight=weights), 1500, watchlist,  maximize=True, verbose_eval=5,

                        feval=gini_xgb, early_stopping_rounds=100)

    resy = pd.DataFrame(model.predict(xgb.DMatrix(X_test)))

    i += 1

    # Saving results for all CV models

    results.append(resy)

    # resy.to_csv(str(i)+'fold.csv')



# Creating the submission file

submission = pd.DataFrame((results[0]+results[1]+results[2]+results[3]+results[4])/5)

#submission.to_csv('sumbission.csv')
from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import SGD

from keras.initializers import random_uniform



import pandas as pd



X_train = pd.read_csv('../input/porto-seguro-safe-driver-prediction/train.csv')

y_train = X_train['target']

X_test = pd.read_csv('../input/porto-seguro-safe-driver-prediction/test.csv')

X_test = X_test.drop(['id'], axis=1)

X_train = X_train.drop(['id', 'target'], axis = 1)

y_train1 = abs(-1+y_train)

y_train = pd.concat([y_train, y_train1], axis=1)

binarizerandscaler = FeatureBinarizatorAndScaler()

binarizerandscaler.fit(X_train)

X_train = binarizerandscaler.transform(X_train)

X_test = binarizerandscaler.transform(X_test)

y_train = y_train.as_matrix()





#hyperparameters

input_dimension = 226

learning_rate = 0.0025

momentum = 0.85

hidden_initializer = random_uniform(seed=SEED)

dropout_rate = 0.2





# create model

model = Sequential()

model.add(Dense(128, input_dim=input_dimension, kernel_initializer=hidden_initializer, activation='relu'))

model.add(Dropout(dropout_rate))

model.add(Dense(64, kernel_initializer=hidden_initializer, activation='relu'))

model.add(Dense(2, kernel_initializer=hidden_initializer, activation='softmax'))



sgd = SGD(lr=learning_rate, momentum=momentum)

model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['acc'])

model.fit(X_train, y_train, epochs=130, batch_size=128)

predictions = model.predict_proba(X_test)



ans = pd.DataFrame(predictions)

ans = ans[0]

#ans.to_csv('./ans.csv', index=False)





    #save the model

model_json = model.to_json()

#with open("./ans.json", "w") as json_file:

#    json_file.write(model_json)



 #   model.save_weights("./ans.h5")
