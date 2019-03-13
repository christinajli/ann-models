import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = open('part3_weight_and_error_kohonen.txt', 'w')
# ------------------------------------------Data pre-processing------------------------------------- #
# import data set
dataSet = pd.read_csv("dataset_noclass.csv")
dataSet = dataSet.iloc[:, :].values


# normalize data to scale inputs to identical ranges
def normalize_data(data_set):
    minmax = [[min(column), max(column)] for column in zip(*data_set)]
    for row in data_set:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    return data_set


dataSet = normalize_data(dataSet)
# --------------------------------Competitive learning ANN model------------------------------------ #
# Global variables
numInputs = 3  # number of input neurons equals number of features, or dimension
numOutputs = 2  # number of output neurons equals number of clusters


def initial_random():
    weights = np.random.uniform(low=-0, high=1, size=(numOutputs, numInputs))
    return weights


def distance(weight, input_data):
    # Euclidean distance
    r = 0
    for i in range(len(weight)):
        r = r + np.square((input_data[i]-weight[i]))
    r = np.sqrt(r)
    return r


def closest_distance(weight_vector, input_vector):
    winning_node = 0
    node = 0
    winning_weights = weight_vector[winning_node]
    closest = distance(winning_weights, input_vector)

    for weights in weight_vector:
        new = distance(weights, input_vector)
        if new < closest:
            closest = new  # update closest distance
            winning_weights = weights
            winning_node = node
        node = node + 1
    return winning_weights, winning_node


def kohonen(data_set, weights):
    """
    Kohonen algorithm pseudo code from textbook
    initialize all weights to random values
    repeat:
        adjust the learning rate
        select an input pattern i_k from the training set
        find node j* whose weight vector w_j is closest to i_k
        update each weight wj*1,...wj*n using the rule:
        delta_wj*,l = learning rate(t)(ik,l - wj*,l) for l in (1...n)
    until network converges or computational bounds are exceeded
    """
    learning_rate = 0.5
    update_weight = 1

    while update_weight >= 0.05:
        learning_rate = learning_rate - 0.05  # adjust learning rate
        for data in data_set:  # select an input pattern i_k
            winning_weight, winning_node = closest_distance(weights, data)
            for i in range(len(winning_weight)):  # update the weights for the winning node
                update_weight = learning_rate * (data[i] - winning_weight[i])
                # inhibitory because only updating the winning node
                weights[winning_node][i] += update_weight
    return weights


def neural_network(data_set, weights):
    output_set = list()
    trained_weights = weights
    current_error = 0
    previous_error = 0
    set1 = list()
    set2 = list()
    while (current_error - previous_error) >= 0:
        error1 = 0
        error2 = 0
        trained_weights = kohonen(data_set, trained_weights)
        for input_data in data_set:
            cluster = closest_distance(weights, input_data)[1]
            output_set.append(cluster)
            if cluster == 0:
                error1 = error1 + np.square(distance(trained_weights, input_data))
                set1.append(list(input_data))
            elif cluster == 1:
                error2 = error2 + np.square(distance(trained_weights, input_data))
                set2.append(list(input_data))
        error1 = np.sum(error1)
        error2 = np.sum(error2)
        previous_error = current_error
        current_error = error1 + error2

    print(previous_error)
    print(trained_weights)

    f.write('Final trained weights: \n')
    f.write(str(trained_weights)+'\n')
    f.write('Sum squared error for cluster one: \n')
    f.write(str(error1)+'\n')
    f.write('Sum squared error for cluster two: \n')
    f.write(str(error2)+'\n')
    f.write('Sum squared error for cluster both: \n')
    f.write(str(previous_error)+'\n')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X1 = list()
    Y1 = list()
    Z1 = list()
    X2 = list()
    Y2 = list()
    Z2 = list()
    for data in set1:
        X1.append(data[0])
        Y1.append(data[1])
        Z1.append(data[2])
    for data in set2:
        X2.append(data[0])
        Y2.append(data[1])
        Z2.append(data[2])
    ax.scatter(X1, Y1, Z1, c = 'r', marker = 'o')
    ax.scatter(X2, Y2, Z2, c='b', marker='o')

    plt.show()
    return output_set


W = initial_random()
f.write("Initial weights: \n")
f.write(str(W)+'\n')
clusters = neural_network(dataSet, W)
print(clusters)
f.write(str(clusters) + '\n')
f.close()

