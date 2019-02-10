"""
The goal of this assignment is to create a single layer perceptron with errors
that predicts classes of wheat using
simple feedback learning,
which is using the correct/incorrect feedback and 
info about (y>=d) or (y<d) to change weights
"""

import csv 
import numpy as np
import random
from numpy import array
from math import exp
#------------------------------------------Initial setup------------------------------------------#
#import data from csv file 
def importData(fileName):
    dataSet = open(fileName).read().split()
    for i in range(len(dataSet)):
        dataSet[i] = dataSet[i].split(',')
        dataSet[i] = [float(dataSet[i][j]) for j in range(len(dataSet[i]))]
        dataSet[i][-1] = int(dataSet[i][-1])
    return dataSet

#normalize dataset
def noramlizeData(dataSet):
    minmax = [[min(column),max(column)] for column in zip(*dataSet)]
    for row in dataSet:
        for i in range(len(row)-1):
            row[i] = (row[i] - minmax[i][0])/(minmax[i][1]-minmax[i][0]) 
    #print(dataSet)
    return dataSet
#----------------------------------------------Perceptron implementation-----------------------------#
#macro definitions
learningRate = 0.01
epoch = 100
weights = []

#initialize random weights for each input
def weightInit (dataSet):
    numOutput = len(set([row[-1] for row in dataSet]))
    numInput = len(dataSet[0])-1

    for i in range(numOutput):
        temp = [] 
        temp.append(0) #introduce bias
        for i in range(numInput):
            temp.append(random.uniform(0, 1))
        weights.append(temp)
    return array(weights) 

#define activation output functions 
def activation (oneSetData,weightsofOneSet): 
    activationValue = 0
    #compute dot product of weights and inputs
    for w in range((len(oneSetData)-1)):
        temp = oneSetData[w] * weightsofOneSet[w]
        activationValue = activationValue + temp
    if (activationValue) >= (weightsofOneSet[0]): 
        return 1
    else:
        return 0

#train the perceptron
def perceptron (trainingData):
    inputs = trainingData[:,:-1]    #first few columns
    outputs = trainingData[:,-1]    #last column is class category

    fixedBias = np.ones((len(inputs),1)) #introduce bias input
    inputs = np.hstack((fixedBias,inputs))
    inputs = array(inputs)

    numInput = len(inputs[0])
    numOutput = len(set([row[-1] for row in trainingData]))

    #class scenarios
    desiredOutput = []
    for row in range(len(outputs)):
        temp = []
        if outputs[row] == 1:
            temp = [1, 0, 0]
        elif outputs [row] == 2:
            temp = [0, 1, 0]
        else:
            temp = [0, 0, 1]
        desiredOutput.append(temp)

    weights = weightInit(trainingData)
    print("Initial weight values:")
    print(weights)

    index = 0
    for _  in range(epoch):
        for j in range (0, numOutput):
            for inputRow in inputs:
                y = activation(inputRow,weights[j])
                #simple feedback learning
                if y > desiredOutput[index][j]:
                    weights[j][index] = weights[j][index] - (learningRate * inputRow[index])
                elif y < desiredOutput[index][j]:
                    weights[j][index] = weights[j][index] + (learningRate * inputRow[index])
                else:
                    index += 1
                if index == numOutput-1:
                    index = 0
    print("Final weight values:")
    print(weights) 
    return weights 

def neuralNetwork (trainingData, testingData):
    predictions = list()
    trainedWeights = perceptron(trainingData)
    
    for row in testingData:
        for i in range(len(set([row[-1] for row in testingData]))): 
            prediction = activation (row,trainedWeights[i])
            if prediction == 1:
                prediction = i + 1
                predictions.append(prediction)
    print(predictions)
    return predictions 
#----------------------------------------------Testing-------------------------------------------#

#evaluate neural network accuracy
def evalNetwork (testingData,predictedValues):
    testOutputs = testingData[:,-1] 
    #predictedOutputs = predictedValues 
    print("see here")
    print(len(predictedValues))
    truePositive  = [0, 0, 0]
    falsePositive = [0, 0, 0]
    falseNegative = [0, 0, 0]
    index = 0
    for row in testOutputs:
        answer = int(row)
        predict = predictedValues[index]
        index += 1
        print('Actual Class:', answer, '; Predicted Class:', predict, end = ' ')
        if answer == predict:
            print('CORRECT!')
            truePositive[answer - 1] += 1
        else:
            print('WRONG :(')
            falsePositive[predict - 1] += 1
            falseNegative[answer - 1] += 1

    numOfClass = len(set(testOutputs))
    for i in range(numOfClass):
        precision = truePositive[i] / (truePositive[i] + falsePositive[i])
        recall = truePositive[i] / (truePositive[i] + falseNegative[i])
        print('Wheat Class' + str(i + 1) + ':')
        print('Precision:' + str(round(precision * 100, 2)) + '%', end = ' ')
        print('Recall:' + str(round(recall * 100, 2)) + '%' + '\n')

#----------------------------------------------Deliverables-------------------------------------------#
"""
1. Predicted output 
2. Initial and final weight values
3. Total number of iterations and terminating condition
4. Precision, recall, and confusion matrix 
"""
trainingSeeds = array(noramlizeData(importData('trainSeeds.csv')))
testingSeeds = array(noramlizeData(importData('testSeeds.csv')))
predictedVal = neuralNetwork (trainingSeeds, testingSeeds)
evalNetwork(testingSeeds,predictedVal)
