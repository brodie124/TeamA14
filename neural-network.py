import random, math

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
    return 1 / (1 + math.exp(x))


def get_sigmoid_derivative(x):
    """

    :param x: float
    :return: float
    """
    return x * (1 - x)


def dot_product(x, y):
    #TODO: implement
    return False


class NeuralNetwork:
    weights = []

    def __init__(self):
        print(self.weights)


    def think(self, inputs):
        """
        Processes inputs
        :param inputs:
        :return: float
        """
        #TODO: implement
        return False

    def generate_weights(self, layers, nodes):
        """
        Generates, and randomizes, the weight list
        with the given amount of layers and amount
        of nodes in each layer.
        :param layers: int, amount of layers in the network
        :param nodes: int, amount of nodes in the network
        :return: list, generated weights
        """
        #TODO: implement
        return False


    def train(self, inputs, outputs, iterations):
        """
        Sets up the synaptic weights for the neural network
        :param inputs: list, normalised training inputs
        :param outputs: list, normalised training outputs
        :param iterations: int, how many times to train
        :return: None
        """
        #TODO: implement
        return False


