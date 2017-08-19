import numpy as np
from data_prep import features_train,features_test,labels_train,labels_test

def sigmoid(x):
    return 1/(1+np.exp(-x))

n_records,n_features=features_train.shape
#weight_list
w=np.random.normal(scale=1/n_features**.5, size=n_features)

#hyper-parameters
iteration=1000
learning_rate=0.5

for i in range(iteration):
    
