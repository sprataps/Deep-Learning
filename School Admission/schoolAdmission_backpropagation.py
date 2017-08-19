import numpy as np
from data_prep import features_train,features_test,labels_train,labels_test

np.randon.seed(21)

def sigmoid(x):
    return 1/(1+np.exp(-x))

#hyper-parameters
learnrate=0.005
iterations=1000
n_hidden=2

n_records,n_features= features_train.shape
lastloss=None

#Initialise weights
weights_input_hidden=np.random.normal(0,scale=1/n_features**.5,size=(n_features,n_hidden))
weights_hidden_output=np.random.normal(0,scale=1/n_features**.5,size=(n_hidden))

for i in range(iterations):
    del_w_input_hidden=np.zeros(weights_input_hidden.shape)
    del_w_hidden_output=np.zeros(weights_hidden_output.shape)
    for x,y in zip(features.values,labels_train):
        ### Forward pass ##
       # Calculate the output
        hidden_input = np.dot(x,weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output = sigmoid(np.dot(hidden_output,weights_hidden_output))
        print(output)
        ## Backward pass ##
        #Calculate the network's prediction error
        error = y-output

        # Calculate error term for the output unit
        output_error_term = error*output*(1-output)
        print("output Error Term: "+str(output_error_term))
        ## propagate errors to hidden layer

        #  Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term,weights_hidden_output)

        # Calculate the error term for the hidden layer
        hidden_error_term = hidden_error*hidden_output*(1-hidden_output)

        #  Update the change in weights
        del_w_hidden_output += learnrate*output_error_term*hidden_output
        del_w_input_hidden += learnrate*hidden_error_term*x[:,None]

    #  Update weights
    weights_input_hidden += del_w_input_hidden/n_records
    weights_hidden_output += del_w_hidden_output/n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
