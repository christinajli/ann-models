"""
Overview: create ANN model using one perceptron to Predict Cardiovascular Disease
"""

import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split

# ------------------------------------------Data pre-processing------------------------------------- #
# import data set
dataSet = pd.read_csv("heart.csv")


# normalize data to scale inputs to identical ranges
def normalize_data(data_set):
    minmax = [[min(column), max(column)] for column in zip(*data_set)]
    for row in data_set:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return data_set


X = dataSet.iloc[:, 0:-1].values
X = normalize_data(X)
y = dataSet.iloc[:, -1].values

# separate into training set, validation set, and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)


# ----------------------------------------------Perceptron NN algorithm----------------------------- #
def activation_function(weights, inputs):
    # activation = sum (weight * input node value)
    activation = inputs.dot(weights)
    return activation


def sigmoid(activation):
    # activation transfer function f(a) = 1/(1+e^-a)
    sigmoid = 1 / (1 + np.exp(-activation))
    out = np.mean(sigmoid)
    return out


def weight_init(num_input, num_output):
    w1 = np.random.uniform(low=0, high=1, size=(num_input, num_output))
    return w1

"""
    Perceptron training algorithm:
    start with randomly chosen weight vector w_0
    let k = 1
    while (there exist input vectors that are misclassified by w_k-1):
        let i_j be a misclassified input vector
        let x_k = class(i_j).i_j, implying that w_k-1.x_k < 0
        update the weight vector to w_k = w_k+1 + nx_k
        increment k

    :param train_input: input from data set
    :param train_output: correct/desired output value, 1 have heart disease, 0 no heart disease
    :return: matrix of weights for each layer
"""

def perceptron(train_input, train_output):

    # constant learning rate to decrease computation expense and avoid being trapped in local minima
    learning_rate = 0.5
    n_epoch = 1

    # introduce bias input, add a column of all 1s
    # fixed_bias = np.ones((len(train_input), 1))
    # train_input = np.hstack((fixed_bias, train_input))

    num_input = len(train_input[0])  # number of input categories
    num_output = len(set(train_output))  # number of output categories

    # start with randomly chosen weight vector w_0
    w1 = weight_init(num_input, num_output)
    print("The initial weights:")
    print(w1)
    print("\n")

    for _ in range(n_epoch):
        for i in range(len(train_input)):
            # input to hidden layer
            sum1 = activation_function(w1, train_input[i])
            predict_out = sigmoid(sum1)

            # matrix different dimension need to reshape
            temp = (learning_rate * train_input[i])
            delta = []
            for row in temp:
                double = [row, row]
                delta.append(double)

            # there exist input vectors that are misclassified by w_k-1
            # update the weight vector to w_k = w_k+1 + nx_k
            if predict_out > train_output[i]:
                w1 = w1 - delta
            elif predict_out < train_output[i]:
                w1 = w1 + delta
    print("The final weights:")
    print(w1)
    print("\n")
    return w1


def neural_network(train_input, train_output, test_input):
    predictions = []
    trained_weights = perceptron(train_input, train_output)

    for row in test_input:
        prediction = activation_function(trained_weights, row)
        prediction = sigmoid(prediction)

        predictions.append(int(round(prediction)))
    print("Predicted values:" + str(predictions))
    return predictions


# ----------------------------------------Perceptron implementation----------------------------------- #
def eval_network(test_output, predicted_output):
    correct = 0
    for i in range(len(test_output)):
        if test_output[i] == predicted_output[i]:
            correct += 1
    accuracy = correct / len(test_output) * 100
    return round(accuracy, 2)


predictedValues = neural_network(X_train, y_train, X_test)
accuracy = eval_network(y_test, predictedValues)
print("Accuracy:" + str(accuracy) + "%")
