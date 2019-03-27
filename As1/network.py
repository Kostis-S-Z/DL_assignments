"""
Created by Kostis S-Z @ 2019-03-27
"""


import numpy as np


class OneLayerNetwork:

    def __init__(self, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """

        var_defaults = {
            "eta": 0.1,
            "n_batch": 10,
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

    def init_weights(self, d, k):
        """
        Initialize a weight matrix of output nodes (k) x features (d) and a bias vector of output nodes x 1
        """
        mean = 0
        std = 0.01

        w = np.random.normal(mean, std, d)

        for i in range(k-1):
            w_i = np.random.normal(mean, std, d)
            w = np.vstack((w, w_i))

        b = np.random.normal(mean, std, k)

        return w, b

    def train(self, data, labels, n_epochs=100, verbose=False):
        """
        Compute forward and backward pass for a number of epochs
        """
        for i in range(n_epochs):

            pred = self.forward(data)

            error = self.backward(pred, labels)

            if verbose:
                self.print_info(i, error)

    def softmax(self, out):
        """
        Softmax activation function
        :return probabilities of the sample being in each class
        """
        return

    def forward(self, data):
        """
        A forward pass in the network computing the predicted class
        :return: a prediction of the class with the highest probability (int:[0,9])
        """
        # calculate the output of the neurons
        sum = np.dot(data, self.w)
        # apply softmax activation function
        p = self.softmax(sum)
        # predicted class is label with highest probability
        k = np.argmax(p)

        return k

    def backward(self, pred):
        """
        A backward pass in the network to compute the error and update the weights
        :param pred: the predictions of the network
        :return: the new weights and the error
        """
        # compute loss function
        error = self.loss(pred)

        # update
        self.w = self.grad_descent()

        return error

    def loss(self):
        """
        Compute the loss function
        :return:
        """
        return

    def grad_descent(self):
        """
        Compute the gradients
        :return:
        """
        return

    def test(self, test_data, test_targets):
        """
        Test a trained model
        """

        pred = self.forward(test_data)

        error = self.misclass_error(pred, test_targets)

        print('Test Error: ', error)
        return error

    def accuracy(self, predictions, targets):
        """
        Percentage of correctly classified predictions
        """
        correct = len(np.where(predictions == targets)[0])
        return float(correct/len(targets))

    def print_info(self, iteration, error):
        print('Iteration: {}'.format(iteration))
        print(' Train Error: {}'.format(error))
        print('\n')
