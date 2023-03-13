import pandas as pd
# import import_ipynb
# import Functions as f
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('bmh')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 20, 10

 'NUM_INSTALMENT_VERSION',
 'NUM_INSTALMENT_NUMBER',
 'Days Extra Taken',
 'AMT_INSTALMENT_difference']
tempinstall=(installments.groupby(['SK_ID_CURR']).mean()).round()
tempinstall['SK_ID_CURR']=tempinstall.index.values
correlation = installment_df.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
from sklearn.preprocessing import Normalizer
normalized_installment = pd.DataFrame(Normalizer().fit_transform(installment_df))
normalized_installment.columns=['SK_ID_CURR',
 'NUM_INSTALMENT_VERSION',
 'NUM_INSTALMENT_NUMBER',
 'Days Extra Taken',
 'AMT_INSTALMENT_difference']
correlation = normalized_installment.corr()
plt.figure(figsize=(15,15))
sns.heatmap(correlation, vmax=1, square=True,annot=True,cmap='cubehelix')
plt.title('Correlation between different fearures')
