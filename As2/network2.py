"""
Created by Kostis S-Z @ 2019-04-03
"""


import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt


class TwoLayerNetwork:

    def __init__(self, **kwargs):
        """
        Initialize Neural Network with data and parameters
        """

        var_defaults = {
            "eta_min": 0.001,  # min learning rate for cycle
            "eta_max": 0.1,  # max learning rate for cycle
            "n_s": 500,  # parameter variable for cyclical learning rate
            "n_batch": 100,  # size of data batches within an epoch
            "n_nodes": 50,  # number of nodes (neurons) in the hidden layer
            "lambda_reg": .1,  # regularizing term variable
            "train_noisy": False,  # variable to toggle adding noise to the training data
            "noise_m": 0,  # the mean of the gaussian noise added to the training data
            "noise_std": 0.01,  # the standard deviation of the gaussian noise added to the training data
            "min_delta": 0.01,  # minimum accepted validation error
            "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
        }

        for var, default in var_defaults.items():
            setattr(self, var, kwargs.get(var, default))

        self.w = []
        self.b = []
        self.models = {}  # Variable to save the weights of the model at the end of each cycle to use it during ensemble
        self.eta = 0.01
        self.p_iter = 0
        self.prev_val_error = 0
        self.loss_train_av_history = []
        self.loss_val_av_history = []

    def init_weights(self, d, k):
        """
        Initialize a weight matrix for the hidden and the output layer
        hidden layer: hidden nodes (m) x features (d) with a bias vector of hidden nodes (m) x 1
        output layer: output nodes (k) x hidden nodes (m) with a bias vector of output nodes (k) x 1
        """
        mean = 0
        std_hidden = 1 / np.sqrt(d)
        std_output = 1 / np.sqrt(self.n_nodes)

        # Initialize hidden layer weights
        hidden_layer = []

        for i in range(self.n_nodes):
            w_i = np.random.normal(mean, std_hidden, d)  # Node i of hidden layer
            hidden_layer.append(w_i)

        # Initialize bias column vector of hidden layer
        hidden_bias = np.zeros(self.n_nodes).reshape(-1, 1)

        # Initialize output layer weights
        output_layer = []

        for i in range(k):
            w_i = np.random.normal(mean, std_output, self.n_nodes)  # Node i of output layer
            output_layer.append(w_i)

        # Initialize bias column vector of output layer
        output_bias = np.zeros(k).reshape(-1, 1)

        self.w.append(np.array(hidden_layer))
        self.w.append(np.array(output_layer))

        self.b.append(np.array(hidden_bias))
        self.b.append(np.array(output_bias))

    def train(self, data, labels, val_data, val_labels, n_epochs=100, early_stop=True, ensemble=False, verbose=False):
        """
        Compute forward and backward pass for a number of epochs
        """
        n = data.shape[0]  # number of samples
        d = data.shape[1]  # number of features
        k = 10  # number of outputs / classes
        indices = np.arange(n)  # a list of the indices of the data to shuffle
        self.init_weights(d, k)

        batch_epochs = int(n / self.n_batch)

        self.loss_train_av_history = []

        for i in range(n_epochs):

            # Shuffle the data and the labels across samples
            np.random.shuffle(indices)  # shuffle the indices and then the data and labels based on this
            data = data[indices]  # current form of data: samples x features
            labels = labels[indices]

            av_acc = 0  # Average epoch accuracy
            av_loss = 0  # Average epoch loss

            self.eta = self.cycle_eta(i)
            print(self.eta)

            for batch in range(batch_epochs):
                start = batch * self.n_batch
                end = start + self.n_batch

                # Number of rows: Features, Number of columns: Samples
                batch_data = data[start:end].T  # Transpose so that features x batch_size
                batch_labels = labels[start:end].T  # Transpose so that classes x batch_size
                batch_classes = np.argmax(batch_labels, axis=0)  # Convert from one-hot to integer form

                if self.train_noisy:
                    batch_data = self.apply_noise(batch_data)

                # Run a forward pass in the network
                # p_output: the result of the softmax function, the real output of the network
                # class_output: by choosing the node with the highest probability we get the predicted class
                hidden_out, prob_out, class_out = self.forward(batch_data)

                # Run a backward pass in the network, computing the loss and updating the weights
                loss = self.backward(hidden_out, batch_data, prob_out, batch_labels)
                av_loss += loss

                av_acc += self.accuracy(class_out, batch_classes)

            average_epoch_loss = av_loss / batch_epochs
            average_epoch_acc = av_acc / batch_epochs
            self.loss_train_av_history.append(average_epoch_loss)

            if verbose:
                print("Epoch: {} - Accuracy: {} Loss: {}".format(i, average_epoch_acc, average_epoch_loss))

            if ensemble:
                if True:  # If the cycle has ended, meaning it the model has reached a local minima TODO: fix condition
                    self.models[i] = [self.w.copy(), self.b.copy()]  # Save weights & bias of the ith cycle  TODO: fix index

            if early_stop:
                val_loss, val_acc = self.test(val_data, val_labels)

                self.loss_val_av_history.append(val_loss)

                val_error = 1 - val_acc
                if self.early_stopping(val_error):
                    break

        if len(self.models) == 0:  # if ensemble was disabled, just save the last model
            self.models[0] = [self.w, self.b]

    def test(self, test_data, test_targets):
        """
        Test a trained model
        """
        n = test_data.shape[0]  # number of samples
        batch_epochs = int(n / self.n_batch)

        models_out = {}
        models_accuracy = {}
        models_loss = {}

        test_labels = np.argmax(test_targets, axis=1)  # Convert one-hot to integer

        # Train each model separately
        for i, model in self.models.items():

            self.w = model[0]  # use the weights of model i
            self.b = model[1]  # use the bias of model i

            model_out = np.zeros(test_labels.shape)

            test_average_loss_i = 0  # Initialize average loss of model i

            for batch in range(batch_epochs):
                start = batch * self.n_batch
                end = start + self.n_batch

                batch_data = test_data[start:end].T
                batch_labels = test_targets[start:end].T

                _, prob_out, class_out = self.forward(batch_data)
                model_out[start:end] = class_out  # Add the batch predictions to the overall predictions

                loss, _ = self.cross_entropy_loss(prob_out, batch_labels)
                test_average_loss_i += loss

            models_loss[i] = test_average_loss_i / batch_epochs
            models_accuracy[i] = self.accuracy(model_out, test_labels)  # Calculate the accuracy of each classifier
            models_out[i] = model_out  # Save the output of the model

            print("Model {} had {}% Test accuracy".format(i, models_accuracy[i] * 100))

        # Concatenate all results to a list
        results = []
        for i, model_results in models_out.items():
            results.append(model_results)

        # Take majority vote across models
        average_out = mode(results, axis=0)[0]

        # Average accuracy over all models
        test_average_acc = self.accuracy(average_out, test_labels)
        # Average loss over all models
        test_average_loss = np.sum(list(models_loss.values())) / batch_epochs

        return test_average_loss, test_average_acc

    def forward(self, data):
        """
        A forward pass in the network computing the predicted class
        """
        # calculate the hidden layer
        s1 = np.dot(self.w[0], data) + self.b[0]
        # apply ReLU activation function
        h = self.relu(s1)
        # calculate the output layer
        s2 = np.dot(self.w[1], h) + self.b[1]
        # apply softmax activation function
        p = self.softmax(s2)
        # predicted class is label with highest probability
        k = np.argmax(p, axis=0)
        return h, p, k

    def backward(self, h, data, p_out, targets):
        """
        A backward pass in the network to update the weights with gradient descent
        """
        # Compute the loss and its gradient
        loss, loss_out_grad = self.cross_entropy_loss(p_out, targets)

        # Add the L2 Regularization term (lambda * ||W||^2) to the loss
        loss = loss + self.reg()

        # Note: In this case (2-layer network) the index 1 and -1 can be used interchangeably
        # Calculate OUTPUT Layer weight and bias gradients
        w_out_grad = np.dot(loss_out_grad, h.T) / self.n_batch
        b_out_grad = np.sum(loss_out_grad, axis=0) / self.n_batch
        # Compute gradient of regularization term w.r.t the OUTPUT weights
        reg_out_grad = 2 * self.lambda_reg * self.w[-1]
        # Update OUTPUT Layer weights and bias
        self.w[-1] = self.w[-1] - self.eta * (w_out_grad + reg_out_grad)
        self.b[-1] = self.b[-1] - self.eta * b_out_grad

        # Calculate HIDDEN Layer loss gradient
        loss_h_grad = np.dot(self.w[-1].T, loss_out_grad)  # Output layer weights x Loss gradient

        indicator = h > 0  # Get a binary representation of values higher than zero (1) or below zero (0)
        loss_h_grad = loss_h_grad * indicator

        # Calculate HIDDEN Layer weight and bias gradients
        w_h_grad = np.dot(loss_h_grad, data.T) / self.n_batch
        b_h_grad = np.sum(loss_h_grad, axis=0) / self.n_batch
        # Compute gradient of regularization term w.r.t the HIDDEN weights
        reg_h_grad = 2 * self.lambda_reg * self.w[0]
        # Update HIDDEN Layer weights and bias
        self.w[0] = self.w[0] - self.eta * (w_h_grad + reg_h_grad)
        self.b[0] = self.b[0] - self.eta * b_h_grad

        return loss

    def softmax(self, out):
        """
        Softmax activation function
        :return probabilities of the sample being in each class
        """
        e_out = np.exp(out - np.max(out))
        return e_out / e_out.sum(axis=0)

    def relu(self, out):
        """
        ReLU activation function
        """
        return np.maximum(0, out)

    def loss(self, p_out, targets):
        """
        Compute the cross-entropy OR the svm multi-class loss
        of a forward pass between the predictions of the network and the real targets
        """
        function = self.loss_function[self.loss_type]
        return function(p_out, targets)

    def cross_entropy_loss(self, p_out, targets):
        """
        Calculate the cross-entropy loss function and its gradient
        """
        # Compute the loss for every class
        loss_batch = - targets * np.log(p_out)
        # Take the mean over samples
        loss_value = np.sum(loss_batch) / self.n_batch

        # Compute the gradient of the loss for the output layer
        loss_grad = - (targets - p_out)

        return loss_value, loss_grad

    def reg(self):
        """
        Compute the regularization term, in this case L2: lambda * ||W||^2
        using the weights of the OUTPUT layer
        """
        return self.lambda_reg * np.sum(np.square(self.w[-1]))

    def apply_noise(self, batch):
        """
        Add small amount of geometric noise to the training images to force the model
        to learn a more general representation of the data
        :return: a noisy batch
        """
        return batch + np.random.normal(self.noise_m, self.noise_std, batch.shape)

    def cycle_eta(self, epoch):
        """
        Calculate the learning rate for a specific cycle
        """
        numer = 1 + epoch
        denom = 2 * self.n_s
        cycle = np.floor(numer / denom)

        part1 = epoch / self.n_s
        part2 = 2 * cycle + 1
        x = np.abs(part1 - part2)

        diff = self.eta_max - self.eta_min
        e_p = diff * np.max(0, (1 - x))

        new_eta = self.eta_min + e_p
        return new_eta

    def early_stopping(self, val_error):
        """
        Early stopping implementation.
        :return: boolean: true if training should stop
        """
        diff = np.abs(val_error - self.prev_val_error)  # If there is a big difference between the validation error
        if diff < self.min_delta:
            self.p_iter += 1
            if self.p_iter > self.patience:
                print("Model reached plateau. Early stopping enabled.")
                return True
        else:
            self.p_iter = 0
        self.prev_val_error = val_error  # Update the previous error to the current one
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

    def plot_loss(self):
        """
        Plot the history of the error
        """
        x_axis = range(1, len(self.loss_train_av_history) + 1)
        y_axis_train = self.loss_train_av_history
        y_axis_val = self.loss_val_av_history
        plt.plot(x_axis, y_axis_train, alpha=0.7, label="Train loss")
        plt.plot(x_axis, y_axis_val, alpha=0.7, label="Validation loss")
        plt.legend(loc='upper right')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()

    def plot_image(self, image, title=""):
        """
        Plot an image with a (optional) title
        """
        image = image.reshape((32, 32, 3), order='F')

        image = (image - image.min()) / (image.max() - image.min())

        image = np.rot90(image, 3)

        plt.imshow(image)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        plt.show()
