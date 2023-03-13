#Import Packages



import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt



from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all' # shows outputs of all commands executed in 1 cell
# Input data files are available in the "../input/" directory.



# List all files under the input directory



input_path = '/kaggle/input/widsdatathon2020'



for dirpath, dirname, filenames in os.walk(input_path):

    for name in filenames:

        print (os.path.join(dirpath , name))

        

# Any results you write to the current directory are saved as output.
# read file

fname = 'training_v2.csv'

train_df = pd.read_csv(os.path.join(input_path , fname))



fname = 'unlabeled.csv'

test_df = pd.read_csv(os.path.join(input_path , fname))



fname = 'solution_template.csv'

solution_df = pd.read_csv(os.path.join(input_path , fname))



print('solution_df')

solution_df.head() 

solution_df.info()

solution_df.shape
solution_df['encounter_id'].describe()
print('test_df')

test_df.head()

test_df.info()

test_df.shape

test_df['encounter_id'].describe()
print('train_df')

train_df.head() 

train_df.info()

train_df.shape
train_df['hospital_death'].dtype

test_df['hospital_death'].dtype

def display_columns_properties(df):

    for i, col in enumerate(df.columns.tolist()):

         print('\n ({} {})  Missing: {}  UniqValsSz: {}'.format(i,col, df[col].isnull().sum() ,df[col].unique().size))

    print('\n')
display_columns_properties(train_df)
display_columns_properties(test_df)
cat_train_df = train_df.select_dtypes(include='object')

cat_train_df.head()

cat_train_df.info()
cat_test_df = test_df.select_dtypes(include='object')

cat_test_df.head()

cat_test_df.info()
def display_columns_uniqvals(df):

    for i, col in enumerate(df.columns.tolist()):

         print('\n ({} {}) Uniq: {} UniqSz: {}'.format(i,col, df[col].unique(),df[col].unique().size ))

    print('\n')
display_columns_uniqvals(cat_test_df)
# Get list of categorical variables

s = (train_df.dtypes == 'object')

object_cols = list(s[s].index)



print("Categorical variables:")

print(object_cols)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Function for comparing different approaches

def score_dataset(X_train, X_valid, y_train, y_valid):

    model = RandomForestRegressor(n_estimators=100, random_state=0)

    model.fit(X_train, y_train)

    preds = model.predict(X_valid)

    return mean_absolute_error(y_valid, preds)
from sklearn.model_selection import train_test_split



# copy the data

train = train_df.copy()





# Select target

y = train['hospital_death']





# To keep things simple, we'll use only numerical predictors

predictors = train.drop(['hospital_death','encounter_id','hospital_id','patient_id','icu_id','readmission_status','ethnicity'], axis=1)







X = predictors.select_dtypes(exclude=['object'])







# Divide data into training and validation subsets

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)



X_train.shape

X_valid.shape



from sklearn.impute import SimpleImputer



# Imputation

my_imputer = SimpleImputer()

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns

'''

from fancyimpute import KNN

knn_imputer = KNN()

imputed_X_train = pd.DataFrame(knn_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(knn_imputer.transform(X_valid))

# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns

'''
'''

from fancyimpute import IterativeImputer

MICE_imputer = IterativeImputer()



imputed_X_train = pd.DataFrame(MICE_imputer.fit_transform(X_train))

imputed_X_valid = pd.DataFrame(MICE_imputer.transform(X_valid))

# Imputation removed column names; put them back

imputed_X_train.columns = X_train.columns

imputed_X_valid.columns = X_valid.columns

'''


#display_columns_properties(imputed_X_train)


from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error





# Define model. Specify a number for random_state to ensure same results each run.

dt_model = DecisionTreeRegressor(random_state=1)



# Fit model using Traing data

dt_model.fit(imputed_X_train, y_train)



# get predicted prices on validation data

predicted_values = dt_model.predict(imputed_X_valid)



# Find difference

score = mean_absolute_error(y_valid, predicted_values)

print('MAE:', score)


test = test_df.copy()



#Separate target

y_test = test['hospital_death']



# To keep things simple, we'll use only numerical predictors



predictors_test = test.drop(['hospital_death','encounter_id','hospital_id','patient_id','icu_id','readmission_status','ethnicity'], axis=1)



X_test = predictors_test.select_dtypes(exclude=['object'])







X_test.shape

X_test.head()


# Imputation

my_imputer = SimpleImputer()

imputed_X_test = pd.DataFrame(my_imputer.fit_transform(X_test))





# Imputation removed column names; put them back

imputed_X_test.columns = X_test.columns

'''

MICE_imputer = IterativeImputer()



imputed_X_test = pd.DataFrame(MICE_imputer.fit_transform(X_test))



# Imputation removed column names; put them back

imputed_X_test.columns = X_test.columns

'''
imputed_X_test.head()


'''

# get predictions on test data

preds = dt_model.predict(imputed_X_test)



# Save predictions in format used for competition scoring

output = pd.DataFrame({'encounter_id': test.encounter_id,

                       'hospital_death': preds},dtype=np.int32)

 

output.to_csv('submission.csv', index=False)

output.head()

'''
### Conclusion

#Used Decision tree model, simple imputation and only numerical columns.
from xgboost import XGBRegressor

X_train = imputed_X_train

X_valid = imputed_X_valid



xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)



xgb_model.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False)



predictions = xgb_model.predict(X_valid)



print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
# get predictions on test data

preds = xgb_model.predict(imputed_X_test)



# Save predictions in format used for competition scoring

output_xgb = pd.DataFrame({'encounter_id': test.encounter_id,

                       'hospital_death': preds},dtype=np.int32)

 

output_xgb.to_csv('submission_xgb1.csv', index=False)

output_xgb.head()