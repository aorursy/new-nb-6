import numpy as np

import pandas as pd



train = pd.read_csv('../input/train.csv')

merchants = pd.read_csv('../input/merchants.csv')

new_merchant_transactions = pd.read_csv('../input/new_merchant_transactions.csv')
new_merchant_transactions.head()
new_merchant_transactions['purchase_amount_integer'] = new_merchant_transactions.purchase_amount.apply(lambda x: x == np.round(x))

print(new_merchant_transactions.groupby('purchase_amount_integer')['card_id'].count())
new_merchant_transactions['purchase_amount_new'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,8)
new_merchant_transactions.purchase_amount_new.head(100)
new_merchant_transactions['purchase_amount_new'] = np.round(new_merchant_transactions['purchase_amount'] / 0.00150265118 + 497.06,2)
new_merchant_transactions['purchase_amount_integer'] = new_merchant_transactions.purchase_amount_new.apply(lambda x: x == np.round(x))

print(new_merchant_transactions.groupby('purchase_amount_integer')['card_id'].count())
new_merchant_transactions.groupby('purchase_amount_new')['card_id'].count().reset_index(name='count').sort_values('count',ascending=False).head(100)
merchants.head(10)
merchants['numerical_1'] = np.round(merchants['numerical_1'] / 0.009914905 + 5.79639, 0)

merchants['numerical_2'] = np.round(merchants['numerical_2'] / 0.009914905 + 5.79639, 0)
merchants.groupby('numerical_1')['merchant_id'].count().head(10)
merchants.groupby('numerical_2')['merchant_id'].count().head(10)
train['target_new'] = 10**(train['target']*np.log10(2))
train.head(10)
train['target_new'].describe()
print(train['target_new'][2], 29/18)

print(train['target_new'][29823], 973/300)