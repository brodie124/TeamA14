import random
import numpy

"""
--- CREDIT ---
NN funtions based on https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
dot product based on http://www.pradeepadiga.me/blog/2017/04/18/finding-dot-product-in-python-without-using-numpy/
"""


def get_sigmoid(x):
    """
    :param x: float
    :return: float
    """
    return 1 / (1 + numpy.exp(-x))


def get_sigmoid_derivative(x):
    """
    :param x: float
    :return: float
    """
    return x * (1 - x)


class NeuralNetwork:
    weights = []

    def __init__(self, inputs_no):
        print(self.weights)
        random.seed(1)
        self.weights = 2 * numpy.random.sample((inputs_no, 1)) - 1

    def think(self, inputs):
        """
        Processes inputs
        :param inputs:
        :return: float
        """
        return get_sigmoid(numpy.dot(inputs, self.weights))

    def train(self, inputs, outputs, iterations):
        for iteration in range(iterations):
            output = self.think(inputs)
            error = outputs - output
            #print("Error:", error)
            adjustment = numpy.dot(inputs.T, error * get_sigmoid_derivative(output))
            #print(adjustment, self.weights)
            #print(iteration)
            self.weights += adjustment
        """
        Sets up the synaptic weights for the neural network
        :param inputs: list, normalised training inputs
        :param outputs: list, normalised training outputs
        :param iterations: int, how many times to train
        :return: None
        """