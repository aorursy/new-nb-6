# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import warnings
warnings.filterwarnings('ignore')
# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
pd.set_option('display.max_columns', None)
df.head()
backup=df.copy()
df.info()
df.describe(include='all')
df['assists']=df['assists'].astype('int16')
df['boosts']=df['boosts'].astype('int16')
df['DBNOs']=df['DBNOs'].astype('int16')
df['headshotKills']=df['headshotKills'].astype('int16')
df['heals']=df['heals'].astype('int16')
df['killPlace']=df['killPlace'].astype('int16')
df['killPoints']=df['killPoints'].astype('int32')
df['kills']=df['kills'].astype('int16')
df['killStreaks']=df['killStreaks'].astype('int16')
df['matchDuration']=df['matchDuration'].astype('int32')
df['maxPlace']=df['maxPlace'].astype('int16')
df['numGroups']=df['numGroups'].astype('int16')
df['rankPoints']=df['rankPoints'].astype('int32')
df['roadKills']=df['roadKills'].astype('int16')
df['teamKills']=df['teamKills'].astype('int16')
df['vehicleDestroys']=df['vehicleDestroys'].astype('int16')
df['weaponsAcquired']=df['weaponsAcquired'].astype('int16')
df['winPoints']=df['winPoints'].astype('int32')
df['winPlacePerc']=df['winPlacePerc'].astype('float32')
df['damageDealt']=df['damageDealt'].astype('float32')
df['longestKill']=df['longestKill'].astype('float32')
df['rideDistance']=df['rideDistance'].astype('float32')
df['swimDistance']=df['swimDistance'].astype('float32')
df['walkDistance']=df['walkDistance'].astype('float32')
df.info()
df['rankPoints'].value_counts()/df.shape[0]*100
df.drop(columns={'rankPoints'},inplace=True)
df.head(1)
df[(df['headshotKills']>40) | (df['heals']>50) | (df['revives']>30) | (df['kills']>60)]
df=df[(df['headshotKills']<25) & (df['revives']<20) & (df['kills']<30)]
df.shape
df.head()
plt.subplots(figsize=(20,7))
sns.heatmap(df.corr())
plt.title('Correlation between the columns')
plt.show()
plt.subplots(figsize=(30,12))
sns.violinplot(x='boosts',y='winPlacePerc',data=df)
plt.title('Boosts vs Win Place')
plt.show()
sns.jointplot(x='walkDistance',y='winPlacePerc',data=df,height=15)
plt.show()
sns.jointplot(x='kills',y='damageDealt',data=df,height=10)
plt.show()
sns.jointplot(x='kills',y='damageDealt',data=df,height=5
              ,kind='hex')
plt.show()
plt.subplots(figsize=(20,10))
sns.boxplot(x='assists',y='winPlacePerc',data=df)
plt.title('Assist vs Win Place')
plt.show()
plt.subplots(figsize=(20,10))
sns.boxplot(x='boosts',y='winPlacePerc',data=df)
plt.title('Boosts vs Win Place')
plt.show()
sns.jointplot(x='damageDealt',y='winPlacePerc',data=df,height=10,kind='hex')
plt.show()
plt.subplots(figsize=(20,10))
sns.barplot(x='DBNOs',y='winPlacePerc',data=df)
plt.title('DBNOs vs Win Place')
plt.show()
plt.subplots(figsize=(20,10))
sns.barplot(x='headshotKills',y='winPlacePerc',data=df)
plt.title('Headshots vs Win Place')
plt.show()
plt.subplots(figsize=(20,10))
sns.boxplot(x='heals',y='winPlacePerc',data=df)
plt.title('Heals vs Win Place')
plt.show()
sns.jointplot(x='killPoints',y='winPlacePerc',data=df,height=10)
plt.show()
plt.subplots(figsize=(20,10))
sns.barplot(x='kills',y='winPlacePerc',data=df)
plt.title('Kills vs Win Place')
plt.show()
plt.subplots(figsize=(20,10))
sns.barplot(x='killStreaks',y='winPlacePerc',data=df)
plt.title('Kill Streaks vs Win Place')
plt.show()
sns.jointplot(x='longestKill',y='winPlacePerc',data=df,height=10)
plt.show()
plt.subplots(figsize=(20,10))
sns.violinplot(x='matchType',y='winPlacePerc',data=df)
plt.title('Match Type vs Win Place')
plt.show()
plt.subplots(figsize=(20,10))
ax=sns.barplot(x='revives',y='winPlacePerc',data=df)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90,ha='right')
plt.title('Revives vs Win Place')
plt.show()
sns.jointplot(x='rideDistance',y='winPlacePerc',data=df,height=15)
plt.show()
plt.subplots(figsize=(20,10))
sns.barplot(x='roadKills',y='winPlacePerc',data=df)
plt.title('Road Kills vs Win Place')
plt.show()
sns.jointplot(x='swimDistance',y='winPlacePerc',data=df,height=10)
plt.show()
plt.subplots(figsize=(20,10))
sns.boxplot(x='teamKills',y='winPlacePerc',data=df)
plt.title('Team Kills vs Win Place')
plt.show()
plt.subplots(figsize=(20,10))
sns.boxplot(x='vehicleDestroys',y='winPlacePerc',data=df)
plt.title('Vehicle Destroys vs Win Place')
plt.show()
sns.jointplot(x='walkDistance',y='winPlacePerc',data=df,kind='hex')
plt.show()
plt.subplots(figsize=(20,10))
sns.barplot(x='weaponsAcquired',y='winPlacePerc',data=df)
plt.title('Weapons Acquired vs Win Place')
plt.show()
sns.jointplot(x='winPoints',y='winPlacePerc',data=df,height=10,kind='hex')
plt.show()
good=df[df['winPlacePerc']>0.6]
good['player_type']=np.where(good['winPlacePerc']>0.9,"Winners","Medium")
sns.relplot(x='walkDistance',y='kills',hue='player_type',data=good,height=7,aspect=3)
plt.title('Walk Distance vs Kills')
plt.show()
sns.relplot(x='headshotKills',y='kills',hue='player_type',data=good,height=7,aspect=3)
plt.title('Headshots vs Kills')
plt.show()
sns.relplot(x='rideDistance',y='walkDistance',hue='player_type',data=good,height=7,aspect=3)
plt.title('Ride Distance vs Walk Distance')
plt.show()
plt.subplots(figsize=(10,10))
ax=sns.countplot(x='matchType',hue='player_type',data=good)
ax.set_xticklabels(ax.get_xticklabels(),rotation=40,ha='right')
plt.title('Frequency of Matches Played')
plt.show()
sns.relplot(x='kills',y='weaponsAcquired',hue='player_type',data=good,height=7,aspect=3)
plt.title('Weapons Acquired vs Kills')
plt.show()
sns.pairplot(good[['kills','walkDistance','rideDistance','matchType','weaponsAcquired','player_type']],hue='player_type',height=5,aspect=2,plot_kws={"s": 5})
plt.show()