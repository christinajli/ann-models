"""
Overview: Competitive K means artificial neural network
"""

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

f = open('part3_weight_and_error_kmeans.txt', 'w')

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
        r = r + np.square((weight[i] - input_data[i]))
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


def centroid(input_data):
    input_sum = [0, 0, 0]
    weights = [0, 0, 0]
    for data in input_data:
        input_sum[0] = input_sum[0] + data[0]
        input_sum[1] = input_sum[1] + data[1]
        input_sum[2] = input_sum[2] + data[2]

    for i in range(len(input_sum)):
        weights[i] = 1/(len(input_data)) * input_sum[i]
    return weights


def error_calculation(data_set, weights):
    error = 0
    for i in range(len(weights)):
        for input_data in data_set:
            error += np.square(distance(weights[i], input_data))
    return error
# -----------------------------k-means clustering algorithm--------------------------------- #
"""
    Kmeans clustering algorithm pseudo code from textbook
    initialize k prototypes (weight vector)
    each cluster Cj is associated with prototype wj
    repeat:
        for each input vector il do:
            place il in the cluster with nearest prototype wj*
        end for
        for each cluster Cj do
            wj = 1/(|cj|) * sum(il) where |cj| is the cluster size
        end for
        compute E = sum of sum (|il - wj|) ^2
    until E no longer decreases, or cluster memberships stabilize
"""

def kmeans (data_set, weights):

    current_error = 0
    epoch = 10
    output_set = list()

    # while (current_error - previous_error) > 0:
    for _ in range(epoch):
        cluster_cj = list()
        for i in range(len(weights)):
            cluster_cj.append(list())

        for i in range(len(data_set)):
            # place data in the cluster with nearest prototype
            winning_node = closest_distance(weights, data_set[i])[1]
            if winning_node == 0:
                cluster_cj[0].append(list(data_set[i]))
            elif winning_node == 1:
                cluster_cj[1].append(list(data_set[i]))

        for i in range(len(weights)):
            # compute centroid from clusters
            weights[i] = centroid(cluster_cj[i])


        error1 = error_calculation(cluster_cj[0], weights)
        error2 = error_calculation(cluster_cj[1], weights)
        current_error = error1 + error2

    f.write('Final trained weights: \n')
    f.write(str(weights) + '\n')
    f.write('Sum squared error for cluster one: \n')
    f.write(str(error1) + '\n')
    f.write('Sum squared error for cluster two: \n')
    f.write(str(error2) + '\n')
    f.write('Sum squared error for cluster both: \n')
    f.write(str(current_error) + '\n')

    for input_data in data_set:
        cluster = closest_distance(weights, input_data)[1]
        output_set.append(cluster)

    return output_set


W = initial_random()
f.write("Initial weights: \n")
f.write(str(W)+'\n')
clusters = kmeans(dataSet, W)
print(clusters)
f.write(str(clusters) + '\n')
f.close()
