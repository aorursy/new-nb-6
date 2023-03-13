# For data manipulation

import pandas as pd              



# Garbage Collector to free up memory

import gc                         

gc.enable()                       # Activate 
orders = pd.read_csv('../input/orders.csv' )

order_products_train = pd.read_csv('../input/order_products__train.csv')

order_products_prior = pd.read_csv('../input/order_products__prior.csv')

products = pd.read_csv('../input/products.csv')

aisles = pd.read_csv('../input/aisles.csv')

departments = pd.read_csv('../input/departments.csv')
'''

#### Remove triple quotes to trim your dataset and experiment with your data

### COMMANDS FOR CODING TESTING - Get 10% of users 

orders = orders.loc[orders.user_id.isin(orders.user_id.drop_duplicates().sample(frac=0.1, random_state=25))] 

'''
orders.head()
order_products_train.head()
order_products_prior.head()
products.head()
aisles.head()
departments.head()
# We convert character variables into category. 

# In Python, a categorical variable is called category and has a fixed number of different values

aisles['aisle'] = aisles['aisle'].astype('category')

departments['department'] = departments['department'].astype('category')

orders['eval_set'] = orders['eval_set'].astype('category')

products['product_name'] = products['product_name'].astype('category')
#Merge the orders DF with order_products_prior by their order_id, keep only these rows with order_id that they are appear on both DFs

op = orders.merge(order_products_prior, on='order_id', how='inner')

op.head()
user= orders.groupby('user_id')[['days_since_prior_order']].mean()

user.columns=['avg_days_since_prior_order']

user.head()


# Create distinct groups for each user, identify the highest order number in each group, save the new column to a DataFrame

user1 = op.groupby('user_id')['order_number'].max().to_frame('u_total_orders')

user1.head()



## Second approach in two steps: 

#1. Save the result as DataFrame with Double brackets --> [[ ]] 

#user = op.groupby('user_id')[['order_number']].max()

#2. Rename the label of the column

#user.columns = ['u_total_orders']

#user.head()
# Reset the index of the DF so to bring user_id from index to column (pre-requisite for step 2.4)

user1 = user1.reset_index()

user1.head()
u_reorder = op.groupby('user_id')['reordered'].mean().to_frame('u_reordered_ratio')

u_reorder = u_reorder.reset_index()

u_reorder.head()
prd = pd.merge(orders, order_products_prior, on='order_id', how='inner')

prd.head(10)
#Getting for each customer, the size of every order they made.

order_size= prd.groupby(['user_id', 'order_id'])['product_id'].count()

order_size = order_size.to_frame()

order_size.columns=['size']

order_size.head()
#Getting the average order size for each user.

results= order_size.groupby('user_id').mean()

results.columns= ['order_size_avg']

results.head()
user = user.merge(u_reorder, on='user_id', how='inner')



del u_reorder

gc.collect()



user.head()
user = user.merge(user1, on='user_id', how='inner')



del user1

gc.collect()



user.head()
user = user.merge(results, on='user_id', how='inner')



del results

gc.collect()



user.head()
# Create distinct groups for each product, count the orders, save the result for each product to a new DataFrame  

prd = op.groupby('product_id')['order_id'].count().to_frame('p_total_purchases')

prd.head()
# Reset the index of the DF so to bring product_id rom index to column (pre-requisite for step 2.4)

prd = prd.reset_index()

prd.head()
# execution time: 25 sec

# the x on lambda function is a temporary variable which represents each group

# shape[0] on a DataFrame returns the number of rows

p_reorder = op.groupby('product_id').filter(lambda x: x.shape[0] >40)

p_reorder.head()
p_reorder = p_reorder.groupby('product_id')['reordered'].mean().to_frame('p_reorder_ratio')

p_reorder = p_reorder.reset_index()

p_reorder.head()
avg_pos = op.groupby('product_id')[['add_to_cart_order']].mean()

avg_pos.columns=['mean_add_to_cart_order']

avg_pos.head()
#Merge the prd DataFrame with reorder

prd = prd.merge(p_reorder, on='product_id', how='left')



#delete the reorder DataFrame

del p_reorder

gc.collect()



prd.head()
prd['p_reorder_ratio'] = prd['p_reorder_ratio'].fillna(value=0)

prd.head()
prd = prd.merge(avg_pos, on='product_id', how='left')



#delete the reorder DataFrame

del avg_pos

gc.collect()



prd.head()
# Create distinct groups for each combination of user and product, count orders, save the result for each user X product to a new DataFrame 

uxp = op.groupby(['user_id', 'product_id'])['order_id'].count().to_frame('uxp_total_bought')

uxp.head()
# Reset the index of the DF so to bring user_id & product_id rom indices to columns (pre-requisite for step 2.4)

uxp = uxp.reset_index()

uxp.head()
times = op.groupby(['user_id', 'product_id'])[['order_id']].count()

times.columns = ['Times_Bought_N']

times.head()
total_orders = op.groupby('user_id')['order_number'].max().to_frame('total_orders')

total_orders.head()
first_order_no = op.groupby(['user_id', 'product_id'])['order_number'].min().to_frame('first_order_number')

first_order_no  = first_order_no.reset_index()

first_order_no.head()
span = pd.merge(total_orders, first_order_no, on='user_id', how='right')

span.head(20)
# The +1 includes in the difference the first order were the product has been purchased

span['Order_Range_D'] = span.total_orders - span.first_order_number + 1

span.head(30)
uxp_ratio = pd.merge(times, span, on=['user_id', 'product_id'], how='left')

uxp_ratio.head()
uxp_ratio['uxp_order_ratio'] = uxp_ratio.Times_Bought_N / uxp_ratio.Order_Range_D

uxp_ratio.head()
uxp_ratio = uxp_ratio.drop(['Times_Bought_N', 'total_orders', 'first_order_number', 'Order_Range_D'], axis=1)

uxp_ratio.head()
#Remove temporary DataFrames

del [times, first_order_number, first_order_no, span]
uxp = uxp.merge(uxp_ratio, on=['user_id', 'product_id'], how='left')



del uxp_ratio

uxp.head()
#Merge uxp features with the user features

#Store the results on a new DataFrame

data = uxp.merge(user, on='user_id', how='left')

data.head()
#Merge uxp & user features (the new DataFrame) with prd features

data = data.merge(prd, on='product_id', how='left')

data.head()
del op, user, prd, uxp

gc.collect()
## First approach:

# In two steps keep only the future orders from all customers: train & test 

orders_future = orders[((orders.eval_set=='train') | (orders.eval_set=='test'))]

orders_future = orders_future[ ['user_id', 'eval_set', 'order_id'] ]

orders_future.head(10)



## Second approach (if you want to test it you have to re-run the notebook):

# In one step keep only the future orders from all customers: train & test 

#orders_future = orders.loc[((orders.eval_set=='train') | (orders.eval_set=='test')), ['user_id', 'eval_set', 'order_id'] ]

#orders_future.head(10)



## Third approach (if you want to test it you have to re-run the notebook):

# In one step exclude all the prior orders so to deal with the future orders from all customers

#orders_future = orders.loc[orders.eval_set!='prior', ['user_id', 'eval_set', 'order_id'] ]

#orders_future.head(10)
# bring the info of the future orders to data DF

data = data.merge(orders_future, on='user_id', how='left')

data.head(10)
#Keep only the customers who we know what they bought in their future order

data_train = data[data.eval_set=='train']

data_train.head()
#Get from order_products_train all the products that the train users bought bought in their future order

data_train = data_train.merge(order_products_train[['product_id','order_id', 'reordered']], on=['product_id','order_id'], how='left' )

data_train.head(15)
#Where the previous merge, left a NaN value on reordered column means that the customers they haven't bought the product. We change the value on them to 0.

data_train['reordered'] = data_train['reordered'].fillna(0)

data_train.head(15)
#We set user_id and product_id as the index of the DF

data_train = data_train.set_index(['user_id', 'product_id'])

data_train.head(15)
#We remove all non-predictor variables

data_train = data_train.drop(['eval_set', 'order_id'], axis=1)

data_train.head(15)
#Keep only the future orders from customers who are labelled as test

data_test = data[data.eval_set=='test']

data_test.head()
#We set user_id and product_id as the index of the DF

data_test = data_test.set_index(['user_id', 'product_id'])

data_test.head()
#We remove all non-predictor variables

data_test = data_test.drop(['eval_set','order_id'], axis=1)

#Check if the data_test DF, has the same number of columns as the data_train DF, excluding the response variable

data_test.head()
# TRAIN FULL 



###########################

## IMPORT REQUIRED PACKAGES

###########################

from sklearn.ensemble import RandomForestClassifier



########################################

## SPLIT DF TO: X_train, y_train (axis=1)

########################################

X_train, y_train = data_train.drop('reordered', axis=1), data_train.reordered



############################

## INITIATE AND TRAIN MODEL

############################

rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1 ,random_state=42)

model = rfc.fit(X_train, y_train)
# TRAIN 80% - VALIDATE 20% 

'''



##########################

##IMPORT REQUIRED PACKAGES

##########################

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split #validate algorithm

from sklearn.metrics import f1_score, classification_report, confusion_matrix



##############################################################

## SPLIT DF TO: 80% for training and 20% as validation (axis=0) 

## & THEN TO to X_train, X_val, y_train, y_val (axis=1)

##############################################################

X_train, X_val, y_train, y_val = train_test_split(data_train.drop('reordered', axis=1), data_train.reordered, test_size=0.8, random_state=42)



##########################

## INITIATE AND TRAIN MODEL

##########################

rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1 ,random_state=42)

model = rfc.fit(X_train, y_train)



#####################################

## SCORE MODEL ON VALIDATION SET (20%)

#####################################

### Predict on validation set with fixed threshold



y_val_pred = (rfc.predict_proba(X_val)[:,1] >= 0.50).astype(int)



### Get scores on validation set



print(f1_score(y_val, y_val_pred, average='binary'))

print(classification_report(y_val, y_val_pred))

print(confusion_matrix(y_val, y_val_pred))



### Remove validate algorithm objects

del [X_val, y_val]



'''
#####################

# FEATURE IMPORTANCE

#####################

feat_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values()

feat_importances.plot(kind='barh')



############################

# DELETE TEMPORARY OBJECTS #

############################

del [X_train, y_train]

gc.collect()
# Predict values for test data with our model from chapter 5 - the results are saved as a Python array

prediction = model.predict(data_test)



## OR Set custom threshold 

prediction = (rfc.predict_proba(data_test)[:,1] >= 0.3).astype(int)
#Save the prediction in a new column in the data_test DF

data_test['prediction'] = prediction

data_test.tail(50)
#Reset the index

final = data_test.reset_index()

#Keep only the required columns to create our submission file (Chapter 6)

final = final[['product_id', 'user_id', 'prediction']]



gc.collect()

final.head()
orders_test = orders.loc[orders.eval_set=='test',("user_id", "order_id") ]

orders_test.head()
final = final.merge(orders_test, on='user_id', how='left')

final.head()
#remove user_id column

final = final.drop('user_id', axis=1)

#convert product_id as integer

final['product_id'] = final.product_id.astype(int)



## Remove all unnecessary objects

del orders

del orders_test

gc.collect()



final.head()
d = dict()

for row in final.itertuples():

    if row.prediction== 1:

        try:

            d[row.order_id] += ' ' + str(row.product_id)

        except:

            d[row.order_id] = str(row.product_id)



for order in final.order_id:

    if order not in d:

        d[order] = 'None'

        

gc.collect()



#We now check how the dictionary were populated (open hidden output)

d
#Convert the dictionary into a DataFrame

sub = pd.DataFrame.from_dict(d, orient='index')



#Reset index

sub.reset_index(inplace=True)

#Set column names

sub.columns = ['order_id', 'products']



sub.head()
#Check if sub file has 75000 predictions

sub.shape[0]

print(sub.shape[0]==75000)
sub.to_csv('sub.csv', index=False)