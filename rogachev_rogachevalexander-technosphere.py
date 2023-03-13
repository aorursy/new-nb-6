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
import sklearn.cross_validation as cv

import sklearn.preprocessing as preprocessing

import sklearn.feature_extraction as fe

import sklearn.ensemble as es
store = pd.read_csv('../input/store.csv')

train = pd.read_csv('../input/train.csv',low_memory=False) #ругался на low_memory

test = pd.read_csv('../input/test.csv')
train.head()
test.fillna(1, inplace=True)

train = train[train["Open"] != 0]
train = pd.merge(train,store,on='Store')

test = pd.merge(test,store,on='Store')
sale_means = train.groupby('Store').mean().Sales

sale_means.name = 'Sales=_Means'



train = train.join(sale_means,on='Store')

test = test.join(sale_means,on='Store')
y = train.Sales.tolist()



train_ = train.drop(['Date','Sales','Store','Customers'],axis=1).fillna(0)



train_dic = train_.fillna(0).to_dict('records')



test_dic = test.drop(["Date","Store","Id"],axis=1).fillna(0).to_dict('records')
dv = fe.DictVectorizer()

X = dv.fit_transform(train_dic)

Xo = dv.transform(test_dic)
maxmin = preprocessing.MinMaxScaler()

X = maxmin.fit_transform(X.toarray())

Xo = maxmin.transform(Xo.toarray())
Xtrain,Xtest,Ytrain,Ytest = cv.train_test_split(X,y)

clf = es.RandomForestRegressor(n_estimators=25)

clf.verbose = True

clf.n_jobs = 8

clf
clf.fit(Xtrain,Ytrain)

print ("Training Score :" + str(clf.score(Xtrain,Ytrain)))

print ("Test Score : " + str(clf.score(Xtest,Ytest)) )
Yresult = clf.predict(Xtest)

Yresult = np.array(Yresult)



Ytest = np.array(Ytest)
result = clf.predict(Xo)

output = pd.DataFrame(test.Id).join(pd.DataFrame(result,columns=['Sales']))

output.to_csv('output.csv',index=False)