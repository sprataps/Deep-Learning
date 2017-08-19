import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data=pd.read_csv('binary.csv')

#adding dummy variables for rank
data=pd.concat([data,pd.get_dummies(data['rank'],prefix='Rank_Number')],axis=1)
data=data.drop('rank',axis=1)

#standardise extra large and extra small features in gre and gpa
for field in ['gre','gpa']:
    mean,std=data[field].mean,data[field].std
    data.loc[:,field]=(data[field]-mean)/std

#split training and testing
labels=data['admit']
data=data.drop('admit',axis=1)
features=data
features_train,labels_train,features_test,labels_test=train_test_split(features,labels,test_size=0.10,random_state=100)
