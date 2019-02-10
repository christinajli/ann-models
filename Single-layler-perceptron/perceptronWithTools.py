import csv 
import numpy as np
from numpy import array
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
#------------------------------------------Built in perceptron-------------------------------------#
trainingData = array(noramlizeData(importData('trainSeeds.csv')))
trainInputs = trainingData[:,:-1]    #first few columns
trainOutputs = trainingData[:,-1]    #last column is class category

testingData = array(noramlizeData(importData('testSeeds.csv')))
testInputs = trainingData[:,:-1]    
testOutputs = trainingData[:,-1]  

#training the sklearn Perceptron 
ppn = Perceptron(n_iter=100, eta0=0.01, random_state=0)
ppn.fit(trainInputs, trainOutputs)
predictedOutputs = ppn.predict(testInputs)
#print (predictedOutputs)
#print('Accuracy: %.2f' % accuracy_score(testOutputs, predictedOutputs))

#----------------------------------------------Deliverables-------------------------------------------#
# Precision and recall values for each class
def evalNetwork (testOutputs,predictedOutputs):
    truePositive  = [0, 0, 0]
    falsePositive = [0, 0, 0]
    trueNegative  = [0, 0, 0]
    falseNegative = [0, 0, 0]
    index = 0
    for row in testOutputs:
        answer = int(row)
        predict = int(predictedOutputs[index])
        index += 1
        if answer == predict:
            truePositive[answer - 1] += 1
        else:
            falsePositive[predict - 1] += 1
            falseNegative[answer - 1] += 1
    numOfClass = len(set(testOutputs))
    for i in range(numOfClass):
        precision = truePositive[i] / (truePositive[i] + falsePositive[i])
        recall = truePositive[i] / (truePositive[i] + falseNegative[i])
        print('Wheat Class' + str(i + 1) + ':')
        print('Precision:' + str(round(precision * 100, 2)) + '%', end = ' ')
        print('Recall:' + str(round(recall * 100, 2)) + '%' + '\n')
evalNetwork(testOutputs,predictedOutputs)
