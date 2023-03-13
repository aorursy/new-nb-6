import pandas as pd               # for data manipulation
import matplotlib.pyplot as plt   # for plotting 
import seaborn as sns             # an extension of matplotlib for statistical graphics
products = pd.read_csv('../input/products.csv')
products.head()
products.shape

products.info()
orders = pd.read_csv('../input/orders.csv' )
orders.head()
orders.days_since_prior_order.value_counts(ascending=True)
plt.figure(figsize=(20,10))

sns.countplot(x="days_since_prior_order", data=orders, color='green')

plt.ylabel('Total Orders')
plt.xlabel('Days since prior order')
plt.title('Days passed since previous order')

plt.show()
order_volume = orders.user_id.value_counts()
order_volume.tail(10)
plt.figure(figsize=(15,5))
graph = sns.countplot(x=order_volume, data=orders, color="Red")
plt.show()
plt.figure(figsize=(15,5))
graph = sns.countplot(x=order_volume, data=orders, palette='Set1')
graph.set( xticks=[0, 96], xticklabels=['4 orders','100 orders'] )
plt.ylabel('Number of Customers')
plt.xlabel('Volume of Orders')
plt.title( 'How many orders do customers make?')
plt.show()