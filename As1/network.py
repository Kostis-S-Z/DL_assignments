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
            "min_delta": 0.01,
            "patience": 10
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.w = None
        self.b = None
        self.p_iter = 0

    def init_weights(self, d, k):
        """
        Initialize a weight matrix of output nodes (k) x features (d) and a bias vector of output nodes x 1
        """
        mean = 0
        std = 0.01

        # Initialize weight matrix
        w = np.random.normal(mean, std, d)

        for i in range(k-1):
            w_i = np.random.normal(mean, std, d)
            w = np.vstack((w, w_i))

        self.w = w

        # Initialize bias vector
        self.b = np.random.normal(mean, std, k)

    def train(self, data, labels, val_data, val_labels, n_epochs=100, early_stop=True, verbose=False):
        """
        Compute forward and backward pass for a number of epochs
        """
        n = data.shape[0]  # number of samples
        d = data.shape[1]  # number of features
        k = 10  # number of outputs / classes
        self.init_weights(d, k)

        batch_epochs = int(n / self.n_batch)

        for i in range(n_epochs):

            np.random.shuffle(data)

            for batch in range(batch_epochs):
                start = batch * self.n_batch
                end = start + self.n_batch

                batch_data = data[start:end]
                batch_labels = labels[start:end]

                pred = self.forward(batch_data)

                # error = self.backward(pred, batch_labels)

            val_acc = self.test(val_data, val_labels)
            val_error = 1-val_acc

            if verbose:
                self.print_info(i, 0, val_error)
            if early_stop and self.early_stopping(val_error):
                print("Model reached plateau. Early stopping enabled.")
                break



    def softmax(self, out):
        """
        Softmax activation function
        :return probabilities of the sample being in each class
        """
        e_out = np.exp(out - np.max(out))
        return e_out / e_out.sum(axis=0)

    def forward(self, data):
        """
        A forward pass in the network computing the predicted class
        :return: a prediction of the class with the highest probability (int:[0,9])
        """
        # calculate the output of the neurons
        s = np.dot(data, np.transpose(self.w)) + self.b
        # apply softmax activation function
        p = self.softmax(s)
        # predicted class is label with highest probability
        k = np.argmax(p, axis=1)

        return k

    def backward(self, pred, targets):
        """
        A backward pass in the network to compute the error and update the weights
        :param pred: the predictions of the network
        :return: the new weights and the error
        """
        # compute loss function
        error = self.loss(pred)

        # update
        #self.w = self.grad_descent()

        return error

    def loss(self, pred):
        """
        Compute the loss function
        :return:
        """
        return 0

    def grad_descent(self):
        """
        Compute the gradients
        :return:
        """
        return

    def early_stopping(self, val_error):
        """
        Early stopping implementation. Depending on the validation error stop or not training
        :return: boolean: true if training should stop
        """
        if val_error < self.min_delta:
            self.p_iter += 1
            if self.p_iter > self.patience:
                return True
        else:
            self.p_iter = 0
        return False

    def test(self, test_data, test_targets):
        """
        Test a trained model
        """

        pred = self.forward(test_data)

        accuracy = self.accuracy(pred, test_targets)

        return accuracy

    def accuracy(self, predictions, targets):
        """
        Percentage of correctly classified predictions
        """
        correct = len(np.where(predictions == targets)[0])
        return float(correct/len(targets))

    def print_info(self, iteration, train_error, val_error):
        print('\n')
        print('Iteration: {}'.format(iteration))
        print(' Train Error: {}'.format(train_error))
        print(' Validation Error: {}'.format(val_error))
