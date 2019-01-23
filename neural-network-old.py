## TODO: Fix mess the resulted from trying to create a multy hidden layered neural network you dumbo


import math
import numpy
from numpy import exp, array, random, dot


##General           :   These are functions that dont fit in anywhere

# Prints a neat table of the an input of list of lists in a +0.000 format
def gPrint(a):
    print(len(a) * "{:+4.3f} ".format(*a))


##TODO: Perhaps refactor, based on article
# Sigmoid function
# x          : Number
def gSigmoid(x):
    return 1 / (1 + math.exp(-x))


# Sigmoid derivative
# x          : Number
def gSigmoidDerivative(x):
    return x * (1 - x)


# Dot product
# x          : List of same length as y
# y          : List of same length as x
# TODO: Figure out how to use zip() in this context
# def dot(x, y):
#     # dproduct = 0
#     # for i in range(0, len(x)) :
#     #    dproduct += x[i]*y[i]
#
#     if type(x) is list and type(y) is not list:
#         return sum([z * y for z in x])
#     elif type(x) is not list and type(y) is list:
#         return sum([z * x for z in y])
#
#     d = 0
#
#     for i, o in zip(x, y):
#         print(i, o)
#         d += i * o
#     return d  # sum(i*o for i,o in zip(x,y))
#
#     ##Test related      :   Test cases here are simply RESULT = [0]


# These are for testing the network functionality, DO NOT use any of these functions
# with actual data.

# Generates a list of lists for test input data
# testcases  : The number of test cases there are
# caseentries: The number of values within each case
def tGenerateIn(testcases, caseentries):
    testin = []
    inputlist = []
    for i in range(0, testcases):
        inputlist.clear()
        for o in range(0, caseentries):
            inputlist.append(random.random())
        testin.append(inputlist[:])
    return testin


# Generates a list of expected outputs for each test case.
# testin     : A list of lists of all test cases, and their values
def tGenerateOut(testin):
    testout = []
    for inputlist in testin:
        testout.append(inputlist[0])
    return array(testout)

    ##Weight related    :   Part of code related to handling weights


# A list of lists [layers][nodes] which holds weights for all nodes.
weights = []


# Generates and randomizes the weight list with given amount of layers and amount of
# nodes in each layer. <<
#               WARNING, THIS CLEARS ANY EXISTING WEIGHT DATA>>
# nodes      : An integer giving the amount of nodes in the network
def wGenerate(nodes):
    nodelist = []
    for o in range(0, nodes):
        nodelist.append(random.uniform(-1, 1))
    return array(nodelist)

    ##Network related   :   Part of code responsible for neural network


##TODO: Perhaps refactor, based on article
# Processes inputs
def nThink(inputs):
    d = dot(inputs, weights)
    return gSigmoid(d)


# Network train sets up the synaptic weights for the neural network.
# trainIn    : A list of lists featuring training inputs    Assumed to be normalized
# trainOut   : A list featuring training outputs            Assumed to be normalized
# iterations : An integer detailing how many times to go
#             Through all training sets.
def nTrain(trainIn, trainOut, iterations):
    global weights
    print("Starting weights: ")
    gPrint(weights)
    for i in range(0, iterations):  # For each iteration
        for o in range(0, len(trainIn)):  # For each sample case
            output = nThink(trainIn[o])
            error = trainOut - output
            adjustment = dot(trainIn.T, error * gSigmoidDerivative(output))
            print("Iteration", i, "Case", o)
            print("Output", output, "Correct output", trainOut[o])
            # gPrint(adjustment)
            gPrint(weights)
            weights += adjustment

    print("Finishing weights: ")
    gPrint(weights)


testin = array(tGenerateIn(20, 10))
weights = wGenerate(len(testin[0]))  # Right now the amount of layers is 1, because I am not mad enough to attempt deep learning for this
nTrain(testin, tGenerateOut(testin).T, 1000)  # and nodes is simply the amount of inputs in each test

# NN funtions based on https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
# dot product based on http://www.pradeepadiga.me/blog/2017/04/18/finding-dot-product-in-python-without-using-numpy/
