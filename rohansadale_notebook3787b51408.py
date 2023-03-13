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
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestRegressor, BaggingRegressor

from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer('english')



df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")

df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")

# df_attr = pd.read_csv('../input/attributes.csv')

df_pro_desc = pd.read_csv('../input/product_descriptions.csv')



num_train = df_train.shape[0]



def str_stemmer(s):

	return " ".join([stemmer.stem(word) for word in s.lower().split()])



def str_common_word(str1, str2):

	words , count = str1.split(),0

	for word in words:

	    if str2.find(word)>0:

	        count+=1

	return count



df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)



df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')



df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))

df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))

df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))



df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)



df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']



df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))

df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))



df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)



df_train = df_all.iloc[:num_train]

df_test = df_all.iloc[num_train:]

id_test = df_test['id']



y_train = df_train['relevance'].values

X_train = df_train.drop(['id','relevance'],axis=1).values

X_test = df_test.drop(['id','relevance'],axis=1).values