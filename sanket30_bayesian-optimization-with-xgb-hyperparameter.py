import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('../input/train.csv',nrows=2_000_000, usecols=[1,2,3,4,5,6,7])
df['pickup_datetime'] = df['pickup_datetime'].str.slice(0, 16)
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')
# Remove observations with missing values
# Since there are only a few of these, i'm not concerned with imputation
df.dropna(how='any', axis='rows', inplace=True)

# Removing observations with erroneous values
mask = df['pickup_longitude'].between(-75, -73)
mask &= df['dropoff_longitude'].between(-75, -73)
mask &= df['pickup_latitude'].between(40, 42)
mask &= df['dropoff_latitude'].between(40, 42)
mask &= df['passenger_count'].between(0, 8)
mask &= df['fare_amount'].between(0, 250)

df = df[mask]
def dist(pickup_lat, pickup_long, dropoff_lat, dropoff_long):  
    distance = np.abs(dropoff_lat - pickup_lat) + np.abs(dropoff_long - pickup_long)
    
    return distance
def transform(data):
    # Extract date attributes and then drop the pickup_datetime column
    data['hour'] = data['pickup_datetime'].dt.hour
    data['day'] = data['pickup_datetime'].dt.day
    data['month'] = data['pickup_datetime'].dt.month
    data['year'] = data['pickup_datetime'].dt.year
    data = data.drop('pickup_datetime', axis=1)

    # Distances to nearby airports, and city center
    # By reporting distances to these points, the model can somewhat triangulate other locations of interest
    nyc = (-74.0063889, 40.7141667)
    jfk = (-73.7822222222, 40.6441666667)
    ewr = (-74.175, 40.69)
    lgr = (-73.87, 40.77)
    data['distance_to_center'] = dist(nyc[1], nyc[0],
                                      data['pickup_latitude'], data['pickup_longitude'])
    data['pickup_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                         data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_jfk'] = dist(jfk[1], jfk[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_ewr'] = dist(ewr[1], ewr[0], 
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_ewr'] = dist(ewr[1], ewr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    data['pickup_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                          data['pickup_latitude'], data['pickup_longitude'])
    data['dropoff_distance_to_lgr'] = dist(lgr[1], lgr[0],
                                           data['dropoff_latitude'], data['dropoff_longitude'])
    
    data['long_dist'] = data['pickup_longitude'] - data['dropoff_longitude']
    data['lat_dist'] = data['pickup_latitude'] - data['dropoff_latitude']
    
    data['dist'] = dist(data['pickup_latitude'], data['pickup_longitude'],
                        data['dropoff_latitude'], data['dropoff_longitude'])
    
    return data


df = transform(df)
import xgboost as xgb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop('fare_amount', axis=1),
                                                    df['fare_amount'], test_size=0.25)
#del(df)
dtrain = xgb.DMatrix(X_train, label=y_train)
#del(X_train)
dtest = xgb.DMatrix(X_test)
#del(X_test)
# def xgb_evaluate(max_depth, gamma,min_child_weight,max_delta_step,subsample,colsample_bytree):
#     params = {'eval_metric': 'rmse',
#               'max_depth': int(max_depth),
#               'subsample': subsample,
#               'eta': 0.1,
#               'gamma': gamma,
#               'colsample_bytree': colsample_bytree,   
#               'min_child_weight': min_child_weight ,
#               'max_delta_step':max_delta_step
#              }
#     # Used around 1000 boosting rounds in the full model
#     cv_result = xgb.cv(params, dtrain, num_boost_round=100, nfold=3)    
    
#     # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
#     return -1.0 * cv_result['test-rmse-mean'].iloc[-1]
# xgb_bo = BayesianOptimization(xgb_evaluate, {
#                                     'max_depth': (2, 12),
#                                      'gamma': (0.001, 10.0),
#                                      'min_child_weight': (0, 20),
#                                      'max_delta_step': (0, 10),
#                                      'subsample': (0.4, 1.0),
#                                      'colsample_bytree' :(0.4, 1.0)})
# # Use the expected improvement acquisition function to handle negative numbers
# # Optimally needs quite a few more initiation points and number of iterations
# xgb_bo.maximize(init_points=3, n_iter=5, acq='ei')
# params = xgb_bo.res['max']['max_params']
# print(params)
params = {'max_depth': 12.0, 'gamma': 0.001, 'min_child_weight': 8.740952582296343, 'max_delta_step': 10.0, 'subsample': 0.4, 'colsample_bytree': 1.0}
params['max_depth'] = int(params['max_depth'])
# Train a new model with the best parameters from the search
model2 = xgb.train(params, dtrain, num_boost_round=250)

# Predict on testing and training set
y_pred = model2.predict(dtest)
y_train_pred = model2.predict(dtrain)

# Report testing and training RMSE
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(np.sqrt(mean_squared_error(y_train, y_train_pred)))
import matplotlib.pyplot as plt
fscores = pd.DataFrame({'X': list(model2.get_fscore().keys()), 'Y': list(model2.get_fscore().values())})
fscores.sort_values(by='Y').plot.bar(x='X')
test = pd.read_csv('../input/test.csv').set_index('key')
test['pickup_datetime'] = test['pickup_datetime'].str.slice(0, 16)
test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'], utc=True, format='%Y-%m-%d %H:%M')

# Predict on holdout set
test = transform(test)
dtest = xgb.DMatrix(test)
y_pred_test = model2.predict(dtest)
holdout = pd.DataFrame({'key': test.index, 'fare_amount': y_pred_test})
holdout.to_csv('submission.csv', index=False)
