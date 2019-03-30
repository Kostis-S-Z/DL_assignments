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
            "eta": 0.1,  # learning rate
            "n_batch": 10,  # size of data batches within an epoch
            "lambda_reg": 0.,  # regularizing term variable
            "min_delta": 0.01,  # minimum accepted validation error
            "patience": 10  # how many epochs to wait before stopping training if the val_error is below min_delta
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

        # Initialize weight matrix classes x features
        w = np.random.normal(mean, std, d)

        for i in range(k-1):
            w_i = np.random.normal(mean, std, d)
            w = np.vstack((w, w_i))

        self.w = w

        # Initialize bias column vector
        self.b = np.random.normal(mean, std, k).reshape(-1, 1)

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

            # Shuffle the data row-wise (across samples)
            np.random.shuffle(data)  # current form of data: samples x features
            av_acc = 0

            for batch in range(batch_epochs):
                start = batch * self.n_batch
                end = start + self.n_batch

                # Number of rows: Features, Number of columns: Samples
                batch_data = data[start:end].T  # Convert so that features x batch_size
                batch_labels = labels[start:end].T  # Convert so that classes x batch_size

                # Run a forward pass in the network
                # p_output: the result of the softmax function, the real output of the network
                # class_output: by choosing the node with the highest probability we get the predicted class
                prob_out, class_out = self.forward(batch_data)

                # Run a backward pass in the network, computing the loss and updating the weights
                loss = self.backward(batch_data, prob_out, batch_labels)

                class_targets = np.argmax(batch_labels, axis=0)
                av_acc += self.accuracy(class_out, class_targets)

            print("Accuracy: {}".format(av_acc / batch_epochs))

            if early_stop:
                val_acc = self.test(val_data, val_labels)
                val_error = 1 - val_acc
                if self.early_stopping(val_error):
                    print("Model reached plateau. Early stopping enabled.")
                    break

    def test(self, test_data, test_targets):
        """
        Test a trained model
        """

        n = test_data.shape[0]  # number of samples
        batch_epochs = int(n / self.n_batch)

        test_average_acc = 0

        for batch in range(batch_epochs):
            start = batch * self.n_batch
            end = start + self.n_batch

            batch_data = test_data[start:end].T
            batch_labels = test_targets[start:end].T

            prob_out, class_out = self.forward(batch_data)

            class_targets = np.argmax(batch_labels, axis=0)

            test_average_acc += self.accuracy(class_out, class_targets)

        return test_average_acc / batch_epochs

    def forward(self, data):
        """
        A forward pass in the network computing the predicted class
        :return: a prediction of the class with the highest probability (int:[0,9])
        """
        # calculate the output of the neurons
        s = np.dot(self.w, data) + self.b
        # apply softmax activation function
        p = self.softmax(s)
        # predicted class is label with highest probability
        k = np.argmax(p, axis=0)

        return p, k

    def softmax(self, out):
        """
        Softmax activation function
        :return probabilities of the sample being in each class
        """
        e_out = np.exp(out - np.max(out))
        return e_out / e_out.sum(axis=0)

    def backward(self, data, p_out, targets):
        """
        A backward pass in the network to update the weights with gradient descent
        """
        # Compute loss function and L2 Regularization term (lambda * ||W||^2)
        loss = self.loss(p_out, targets) + self.reg()

        # Compute the gradient of the loss
        loss_grad = - (targets - p_out)
        # Compute the gradient w.r.t the weights
        #   -> inner product (sum) of the loss*data_inputs
        w_grad = np.dot(loss_grad, data.T)

        # Compute the gradient w.r.t the bias
        #   -> inner product (sum) of the loss*bias_inputs where bias_inputs is a vector of 1s, each for every sample
        #   -> its basically the same as doing np.sum(loss_grad, axis=1)
        b_grad = np.sum(loss_grad, axis=0)  # np.dot(loss_grad, np.ones((self.n_batch, 1)))

        # Divide with the size of the batch
        w_grad = w_grad / self.n_batch
        b_grad = b_grad / self.n_batch

        # Compute gradient of regularization term w.r.t the weights
        reg_grad = 2 * self.lambda_reg * self.w

        # Update weights and bias
        self.w = self.w - self.eta * (w_grad + reg_grad)
        self.b = self.b - self.eta * b_grad

        return loss

    def loss(self, p_out, targets):
        """
        Compute the cross-entropy loss of a forward pass between the predictions of the network and the real targets
        """
        loss_batch = - np.log(np.dot(targets.T, p_out))
        return loss_batch.sum(axis=0) / self.n_batch

    def reg(self):
        """
        Compute the regularization term, in this case L2: lambda * ||W||^2
        """
        return self.lambda_reg * np.sum(np.square(self.w))

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
