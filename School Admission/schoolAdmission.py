import numpy as np
from data_prep import features_train,features_test,labels_train,labels_test

def sigmoid(x):
    return 1/(1+np.exp(-x))

n_records,n_features=features_train.shape
#weight_list
weights=np.random.normal(scale=1/n_features**.5, size=n_features)

#hyper-parameters
iteration=1000
learning_rate=0.5

for i in range(iteration):
    delta_weight=np.zeros(n_features)
    for x,y in zip(features_train,labels_train):
        output=sigmoid(np.dot(weights,x))
        #calculate the error y-y-hat
        error=y-output
        '''
        calculate the weight step i.e learning_rate*delta*xi
        delta=error*d(output)/d(wi)i.e error*d(sigmoid(x))/d(wi)
        '''
        #calculate error term
        err_term=error*output*(1-output)

        #calculate increase in weight with each feature
        delta_weight+= err_term*x
    weights+= learning_rate*delta_weight/n_features

#calculate accuracy on the test data
test_output=sigmoid(weights,features_test)
predictions=test_output>0.5
accuracy=accuracy_score(labels_test,predictions)
print("Accuracy: {:.3f}".format(accuracy))
    
