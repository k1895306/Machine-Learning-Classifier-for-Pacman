# classifierAgents.py
# parsons/07-oct-2017
#
# Version 1.0
#
# Some simple agents to work with the PacMan AI projects from:
#
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agents here are extensions written by Simon Parsons, based on the code in
# pacmanAgents.py

from pacman import Directions
from game import Agent
import api
import random
import game
import util
import sys
import os
import csv
import numpy as np
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import operator

# Here, I have only used numpy and sklearn packages to build my classifier.

# ClassifierAgent
#
# An agent that runs a classifier to decide what to do.
class ClassifierAgent(Agent):

    # Constructor. This gets run when the agent starts up.
    def __init__(self):
        print("Initialising")

    # Take a string of digits and convert to an array of
    # numbers. Exploits the fact that we know the digits are in the
    # range 0-4.
    #
    # There are undoubtedly more elegant and general ways to do this,
    # exploiting ASCII codes.
    def convertToArray(self, numberString):
        numberArray = []
        for i in range(len(numberString) - 1):
            if numberString[i] == '0':
                numberArray.append(0)
            elif numberString[i] == '1':
                numberArray.append(1)
            elif numberString[i] == '2':
                numberArray.append(2)
            elif numberString[i] == '3':
                numberArray.append(3)
            elif numberString[i] == '4':
                numberArray.append(4)

        return numberArray

    # This is my implementation of K-fold Cross Validation specific for my NB classifier
    # This is for MODEL ACCURACY TESTING
    def kFoldValidation(self, data, target, k=7):
        # print("enters K Fold Validation")
        # Test Score is to get the list of scores of the test set classifier 
        testScore = []
        shuffleData = []
        newData = [i for i in range(len(data))]
        shuffleTarget = []

        # inserting the data and then replacing the data with the shuffle Data
        for i in newData:
            shuffleData.append(data[i])
            shuffleTarget.append(target[i])

        data = shuffleData
        target = shuffleTarget
        # using list to split data for training and testing
        for i in range(k):
            splitTarget = [target[t::k] for t in range(k)]
            splitData = [data[d::k] for d in range(k)]
            valTarget = splitTarget[i]
            valData = splitData[i]
            splitData.pop(i)
            splitTarget.pop(i)
            learningData = np.concatenate(splitData)
            learningTarget = np.concatenate(splitTarget)

            # learns the classifier from the learningData and from the test set, we get the classifierScore
            self.probList, self.priorProbability = self.NBTraining(learningData, learningTarget)
            classifierScore = self.bayesScore(valData, valTarget)[0]
            testScore.append(classifierScore)

        # We use the numpy package to get the mean of the K and return the value.
        kArray = np.asarray(testScore)
        kResults = kArray.mean()
        return kResults

    # Training and Testing Splitter for score split validation.
    # It's to train and test the classifier.
    def scoreSplitValidation(self, data, target, split):
        # print("enters trainingTestingSplitter")

        # Shuffling the data, adding them and replacing the original data
        shuffleTarget = []
        shuffleData = []
        newData = [d for d in range(len(data))]
        for i in newData:
            shuffleData.append(data[i])
            shuffleTarget.append(target[i])
        data = shuffleData
        target = shuffleTarget

        # first, computing the length of the data and then spliting the data into test and training. 
        dataLength = int(len(data)) * float(split)
        dataLength = int(dataLength)
        # Testing and Training data
        xTesting = data[:dataLength]
        yTesting = target[:dataLength]
        xTraining = data[dataLength + 1:]
        yTraining = target[dataLength + 1:]

        # Runs the classifier to train the training data.
        self.probList, self.priorProbability = self.NBTraining(xTraining, yTraining)
        # we use the bayesScore function to calculate the accuracy
        returnScore = self.bayesScore(xTesting, yTesting)[0]
        return returnScore

    # NB TRAINING AND CLASSIFIER
    # I have created a NB classifier and it has been compared with other (scikit learn) classifiers using
    # the 7-fold cross validation (function - kFoldValidation) and as a result, I observed that NB has a high degree of accuracy.

    def NBTraining(self, data, target):
        # print("enters NBTraining")
        # Calculating for each class to get the prior Probability and compute the probability
        # List.
        # For calculating the probability list, we need to create a dictionary so we can split the data.
        idx = 0
        # all initial prior probability is set to 0
        priorProbability = [0, 0, 0, 0]
        for t in target:
            if t == 0:
                priorProbability[0] = priorProbability[0] + 1
            if t == 1:
                priorProbability[1] = priorProbability[1] + 1
            if t == 2:
                priorProbability[2] = priorProbability[2] + 1
            if t == 3:
                priorProbability[3] = priorProbability[3] + 1

        # Spliting the data with a dictionary
        splitData = {0: [], 1: [], 2: [], 3: []}
        for t in target:
            if t == 0:
                splitData[0].append(data[idx])
            if t == 1:
                splitData[1].append(data[idx])
            if t == 2:
                splitData[2].append(data[idx])
            if t == 3:
                splitData[3].append(data[idx])
            idx += 1

        # I've created a list of probability dictionary,so we find the prob = (1) for a feature
        # We then calculate the no of instances for the attribute with 1
        probs = {0: [], 1: [], 2: [], 3: []}
        for k, val in splitData.iteritems():
            valArray = np.asarray(val)
            arraySum = valArray.sum(axis=0)
            for i in arraySum:
                # computing the probability of getting 1 for feature
                # laplace smooth
                probOne = float(i + 1) / float(len(splitData[k]) + 25)
                # appending the probability of 1 for every feature in the class
                probs[k].append(probOne)
        # We return the list of probabilities and the prior Probability - this list is the trained data.
        # simple enumeration
        return probs, priorProbability

    # NB Testing
    def NBTesting(self, features):
        # print("enters NBTesting")
        # calculating the probability
        output = {}
        for key in range(len(self.probList)):

            # empty list for probability values
            probabilityValue = []
            # computing the prior Probability
            priorSum = sum(self.priorProbability)
            priorProb = float(self.priorProbability[key]) / float(priorSum)
            # adding it to the probability values
            probabilityValue.append(priorProb)

            # looping over every feature to calc the probability(1) for every key(v).
            for v in range(len(features)):

                # if the probability != 1 then we compute the probability as 1 - probability of it being 1
                # else if the probability = 1 then the probability based on the trained data is calculated.
                if features[v] == 0:
                    probabilityValue.append(1 - float(self.probList[key][v]))
                else:
                    probabilityValue.append(float(self.probList[key][v]))
            # store the probability values for the features and for the class.
            probArray = np.array(probabilityValue)
            probabilityValue = np.prod(probArray)
            output[key] = probabilityValue
        # Return the maximum probability output as a number
        return max(output.iteritems(), key=operator.itemgetter(1))[0]

    # Computing the Confusion Metric
    def bayesScore(self, data, target):
        # print("enters bayesScore")
        # creating an empty matrix and an empty list to hold the results
        confusionMatrix = np.zeros((4, 4))
        accuracyScore = 0
        # results for every data set row
        results = []
        for i in range(len(data)):
            rowResult = self.NBTesting(data[i])
            results.append(rowResult)
        # we are comparing the result with the target and computing the accuracy score
        for x, y in zip(results, target):
            confusionMatrix[x, y] = confusionMatrix[x, y] + 1
        for i in range(confusionMatrix[0].shape[0]):
            accuracyScore = accuracyScore + confusionMatrix[i, i]
        matrixSum = confusionMatrix.sum()
        results = float(accuracyScore) / float(matrixSum)
        return results, confusionMatrix

    # This gets run on startup. Has access to state information.
    #
    # Here we use it to load the training data.
    def registerInitialState(self, state):

        # open datafile, extract content into an array, and close.
        self.datafile = open('good-moves.txt', 'r')
        content = self.datafile.readlines()
        self.datafile.close()

        # Now extract data, which is in the form of strings, into an
        # array of numbers, and separate into matched data and target
        # variables.
        self.data = []
        self.target = []
        # Turn content into nested lists
        for i in range(len(content)):
            lineAsArray = self.convertToArray(content[i])
            dataline = []
            for j in range(len(lineAsArray) - 1):
                dataline.append(lineAsArray[j])

            self.data.append(dataline)
            targetIndex = len(lineAsArray) - 1
            self.target.append(lineAsArray[targetIndex])

        # data and target are both arrays of arbitrary length.
        #
        # data is an array of arrays of integers (0 or 1) indicating state.
        #
        # target is an array of imtegers 0-3 indicating the action
        # taken in that state.

        # *********************************************
        #
        # Any other code you want to run on startup goes here.
        #
        # I'm running the classifier here , it returns the score
        # *********************************************

        # with the function kFoldValidation for cross validation gives the NBayes classifier score.
        self.crossValidationScore = self.kFoldValidation(self.data, self.target, 7)

        # with 0.2 split I'm training and testing for my own classifier.
        self.splitScore = self.scoreSplitValidation(self.data, self.target, 0.2)

        # data Leaning classifier
        self.probList, self.priorProbability = self.NBTraining(self.data, self.target)

        # computing the confusion matrix and training score
        self.trainingScore, self.confusionMatrix = self.bayesScore(self.data, self.target)

        # We want to print metrics only once
        self.score = 0

        # Using scikit learn metrics to compare algorithms
        clf = BernoulliNB().fit(self.data, self.target)
        clf = BernoulliNB().fit(self.data, self.target)
        self.scikitTrainingScore = clf.score(self.data, self.target)
        self.scikitCrossValidationScore = cross_val_score(clf, self.data, self.target, cv=7).mean()
        self.scikitConfusionMatrix = confusion_matrix(self.target, clf.predict(self.data))

    # Tidy up when Pacman dies
    def final(self, state):

        print("I'm done!")

    # Turn the numbers from the feature set into actions:
    def convertNumberToMove(self, number):
        if number == 0:
            return Directions.NORTH
        elif number == 1:
            return Directions.EAST
        elif number == 2:
            return Directions.SOUTH
        elif number == 3:
            return Directions.WEST

    # Here we just run the classifier to decide what to do
    def getAction(self, state):
        # print("enters getAction")

        # How we access the features.
        features = api.getFeatureVector(state)

        # Calculating learning metrics for fine tuning of hyperparameters (if needed).
        # print("self.score:" + str(self.score))

        if self.score == 0:
            print \
                ("Full Data set training accuracy: %.3f" % (self.trainingScore))
            print \
                ("*************************************************************\n")
            print \
                ("0.2 Split Validation accuracy: %.3f" % (self.splitScore))
            print \
                ("*************************************************************\n")
            print \
                ("Full Data set 7-fold cross validation score: %.3f" % (self.crossValidationScore))
            print \
                ("*************************************************************\n")
            print \
                ("My Confusion Matrix: \n %s" % (self.confusionMatrix))
            print \
                ("*************************************************************\n")

            # comparing with other classifiers
            print \
                "Training accuracy for scikit: %.3f" % (self.scikitTrainingScore)
            print \
                ("*************************************************************\n")
            print \
                "Scikit 7-fold cross validation score: %.3f" % (self.scikitCrossValidationScore)
            print \
                ("*************************************************************\n")
            print \
                "Scikit confusion matrix: \n%s\n" % (self.scikitConfusionMatrix)
            self.score = 1

        # Using NB to classify feature vector
        actionNum = self.NBTesting(features)
        # action number to a move
        action = self.convertNumberToMove(actionNum)

        # Get the actions we can try.
        legal = api.legalActions(state)

        # getAction has to return a move. Here we ask the
        # API to ask Pacman to make the best move if the move is in legal.
        # If not Pacman makes a random choice of action.
        # return api.makeMove(Directions.STOP, legal)
        if action in legal:
            return api.makeMove(action, legal)
        else:
            return api.makeMove(random.choice(legal), legal)
