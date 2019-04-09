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
            "loss_type": "svm",  # cross-entropy or svm
            "svm_margin": 1.,  # margin parameter for svm loss
            "lambda_reg": .1,  # regularizing term variable
            "train_noisy": False,  # variable to toggle adding noise to the training data
            "noise_m": 0,  # the mean of the gaussian noise added to the training data
            "noise_std": 0.01,  # the standard deviation of the gaussian noise added to the training data
            "min_delta": 0.01,  # minimum accepted validation error
            "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
        }
        self.loss_function = {
            "cross-entropy": self.cross_entropy,
            "svm": self.svm_multi
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
                real_out, prob_out, class_out = self.forward(batch_data)
                # In the case of the SVM loss you update the weights with the unfiltered output
                if self.loss_type == "svm":
                    prob_out = real_out
                # Run a backward pass in the network, computing the loss and updating the weights
                loss = self.backward(batch_data, prob_out, batch_labels)
                av_loss += loss

                av_acc += self.accuracy(class_out, batch_classes)

            average_epoch_loss = av_loss / batch_epochs
            average_epoch_acc = av_acc / batch_epochs
            self.loss_train_av_history.append(average_epoch_loss)

            if verbose:
                print("Epoch: {} - Accuracy: {} Loss: {}".format(i, average_epoch_acc, average_epoch_loss))

            if ensemble:
                if True: # If the cycle has ended, meaning it the model has reached a local minima TODO: fix condition
                    self.models[i] = [self.w, self.b]  # Save weights & bias of the ith cycle  TODO: fix index

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

                real_out, prob_out, class_out = self.forward(batch_data)
                model_out[start:end] = class_out  # Add the batch predictions to the overall predictions

                # In the case of the SVM loss you update the weights with the unfiltered output
                if self.loss_type == "svm":
                    prob_out = real_out

                loss, _ = self.loss(prob_out, batch_labels)
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
        s = np.dot(self.w[1], h) + self.b[1]
        # apply softmax activation function
        p = self.softmax(s)
        # predicted class is label with highest probability
        k = np.argmax(p, axis=0)
        return s, p, k

    def backward(self, data, p_out, targets):
        """
        A backward pass in the network to update the weights with gradient descent
        """
        # Compute the loss and its gradient
        loss, loss_grad = self.loss(p_out, targets)

        # Add the L2 Regularization term (lambda * ||W||^2) to the loss
        loss = loss + self.reg()

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

    def cross_entropy(self, p_out, targets):
        """
        Calculate the cross-entropy loss function and its gradient
        """
        # Compute the loss for every class
        loss_batch = - targets * np.log(p_out)
        # Take the mean over samples
        loss_value = np.sum(loss_batch) / self.n_batch

        # Compute the gradient of the loss
        loss_grad = - (targets - p_out)
        return loss_value, loss_grad

    def svm_multi(self, p_out, targets):
        """
        Calculate the SVM multi-class loss function and its gradient
        """
        # Convert the one-hot back to integer representation
        targets_class = np.argmax(targets, axis=0)
        # Get list of indices for the batch samples
        indices = np.arange(self.n_batch)
        # Transpose the output matrix to get the correct dimensions
        p_out_t = p_out.T
        # For every batch get its score (probability) for the actual (target) class it belongs to
        correct_p = p_out_t[indices, targets_class]
        # For every sample prediction in the batch
        # subtract from each class the number of prediction of the correct class, phew that's a mouthful...
        sub_p = p_out - correct_p
        # Add a margin parameter
        sub_p = sub_p + self.svm_margin
        # Replace negative values with zero
        margins = np.maximum(0, sub_p)
        # Transpose the margins matrix to get the correct dimensions
        margins = margins.T
        # Replace the correct class predictions with zero
        margins[indices, targets_class] = 0
        # Sum over classes
        sum = np.sum(margins, axis=1)
        # Take the mean loss over the batch samples
        loss_value = np.mean(sum)

        # Compute the gradient SVM loss
        margins_b = margins
        # Convert to binary values: 1 -> weights need to updated (wrong predictions) 0 -> weights not to update
        margins_b[margins > 0] = 1
        # Sum over the samples for every class
        wrongs_per_class = np.sum(margins_b, axis=1)
        # Convert the matrix and the vector to correct forms
        margins_b = margins_b
        # At each class subtract the number of the wrongly predicted samples
        margins_b[indices, targets_class] = - wrongs_per_class
        # Transpose it back
        loss_grad = margins_b.T
        return loss_value, loss_grad

    def reg(self):
        """
        Compute the regularization term, in this case L2: lambda * ||W||^2
        """
        return self.lambda_reg * np.sum(np.square(self.w))

    def apply_noise(self, batch):
        """
        Add small amount of geometric noise to the training images to force the model
        to learn a more general representation of the data
        :return: a noisy batch
        """
        return batch + np.random.normal(self.noise_m, self.noise_std, batch.shape)

    def cycle_eta(self, cycle, l):
        """
        Calculate the learning rate for a specific cycle
        """
        #TODO: Finish this
        numer = cycle - (2 * l * self.n_s)
        diff = self.eta_max - self.eta_min
        new_eta = self.eta_min + (numer / self.n_s) * diff
        self.eta = new_eta

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
