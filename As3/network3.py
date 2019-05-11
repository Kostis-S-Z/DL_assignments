"""
Created by Kostis S-Z @ 2019-04-03
"""

import copy
import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt

from batch_norm import BatchNormalization


class MultiLayerNetwork:

    def __init__(self, **kwargs):
        """
        Initialize a Multi-Layer Neural Network with parameters
        """

        var_defaults = {
            "eta_min": 1e-5,  # min learning rate for cycle
            "eta_max": 1e-1,  # max learning rate for cycle
            "n_s": 500,  # parameter variable for cyclical learning rate
            "n_batch": 100,  # size of data batches within an epoch
            "lambda_reg": .1,  # regularizing term variable
            "init_type": "Xavier",  # Choose between Xavier and He initialisation
            "dropout": False,  # Use dropout or not
            "dropout_perc": 0.2,  # Percentage of nodes to dropout
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
        self.batch_norm = None
        self.models = {}  # Variable to save the weights of the model at the end of each cycle to use it during ensemble
        self.p_iter = 0
        self.eta = 0.01
        self.prev_val_error = 0
        self.loss_train_av_history = []
        self.cost_train_av_history = []
        self.acc_train_av_history = []
        self.loss_val_av_history = []
        self.cost_val_av_history = []
        self.acc_val_av_history = []
        self.eta_history = []

    def init_weights(self, net_structure, d):
        """
        Initialize a weight matrix for the hidden and the output layers
        """
        mean = 0
        if self.init_type == "Xavier":
            numer = 1
        else:  # use He initialisation
            numer = 2

        dim_prev_layer = d
        # For every layer (including the output)
        for l in range(len(net_structure)):
            # Calculate standard deviation to initialise the weights of layer i
            std = numer / np.sqrt(dim_prev_layer)
            # Initialize weight matrix of layer i
            w_layer_i = np.random.normal(mean, std, (net_structure[l], dim_prev_layer))
            # Initialize bias column vector of layer i
            hidden_bias = np.zeros(net_structure[l]).reshape(-1, 1)

            # The second dimension of the weight matrix of the next layer is the number of nodes of the current one
            dim_prev_layer = net_structure[l]
            # Add weights & bias to network
            self.w.append(np.array(w_layer_i))
            self.b.append(np.array(hidden_bias))

    def train(self, network_structure, data, labels, val_data, val_labels,
              n_epochs=100, use_batch_norm=False, early_stop=False, ensemble=False, verbose=False):
        """
        Compute forward and backward pass for a number of epochs
        """
        n = data.shape[0]  # number of samples
        d = data.shape[1]  # number of features
        indices = np.arange(n)  # a list of the indices of the data to shuffle

        self.init_weights(network_structure, d)

        if use_batch_norm:
            self.batch_norm = BatchNormalization(network_structure, self.n_batch)

        batch_epochs = int(n / self.n_batch)

        self.loss_train_av_history = []
        self.cost_train_av_history = []
        self.acc_train_av_history = []
        self.eta_history = []

        iteration = 0

        for i in range(n_epochs):

            # Shuffle the data and the labels across samples
            np.random.shuffle(indices)  # shuffle the indices and then the data and labels based on this
            data = data[indices]  # current form of data: samples x features
            labels = labels[indices]

            av_acc = 0  # Average epoch accuracy
            av_loss = 0  # Average epoch loss
            av_cost = 0  # Average epoch cost

            for batch in range(batch_epochs):
                # Calculate a new learning rate based on the CLR method
                self.eta = self.cycle_eta(iteration)
                self.eta_history.append(self.eta)
                iteration += 1

                start = batch * self.n_batch
                end = start + self.n_batch

                # Number of rows: Features, Number of columns: Samples
                batch_data = data[start:end].T  # Transpose so that features x batch_size
                batch_labels = labels[start:end].T  # Transpose so that classes x batch_size
                batch_classes = np.argmax(batch_labels, axis=0)  # Convert from one-hot to integer form

                if self.train_noisy:
                    batch_data = self.apply_noise(batch_data)

                # Run a forward pass in the network
                # layers_out: the output of each layer
                # class_output: by choosing the node with the highest probability we get the predicted class
                layers_out, class_out = self.forward(batch_data)

                # Run a backward pass in the network, computing the loss and updating the weights
                loss, cost, _, _ = self.backward(layers_out, batch_data, batch_labels)
                av_loss += loss
                av_cost += cost

                av_acc += self.accuracy(class_out, batch_classes)

                if ensemble:
                    # If the cycle has ended, the learning rate will be at its lowest
                    # meaning it the model has reached a local minima
                    if self.eta == self.eta_min:
                        # Save weights & bias of the ith cycle
                        self.models[i] = [self.w.copy(), self.b.copy()]

            average_epoch_loss = av_loss / batch_epochs
            average_epoch_cost = av_cost / batch_epochs
            average_epoch_acc = av_acc / batch_epochs
            self.loss_train_av_history.append(average_epoch_loss)
            self.cost_train_av_history.append(average_epoch_cost)
            self.acc_train_av_history.append(average_epoch_acc)

            if verbose:
                print("Epoch: {} - Accuracy: {} Loss: {}".format(i, average_epoch_acc, average_epoch_loss))

            if not ensemble:
                self.models[0] = [self.w, self.b]  # if ensemble was disabled, just save the last model

            val_loss, val_cost, val_acc = self.test(val_data, val_labels)

            self.loss_val_av_history.append(val_loss)
            self.cost_val_av_history.append(val_cost)
            self.acc_val_av_history.append(val_acc)

            if early_stop:
                val_error = 1 - val_acc
                if self.early_stopping(val_error):
                    break

    def test(self, test_data, test_targets):
        """
        Test a trained model
        """
        n = test_data.shape[0]  # number of samples
        batch_epochs = int(n / self.n_batch)

        models_out = {}
        models_accuracy = {}
        models_loss = {}
        models_cost = {}

        test_labels = np.argmax(test_targets, axis=1)  # Convert one-hot to integer

        # Train each model separately
        for i, model in self.models.items():

            self.w = model[0]  # use the weights of model i
            self.b = model[1]  # use the bias of model i

            model_out = np.zeros(test_labels.shape)

            test_average_loss_i = 0  # Initialize average loss of model i
            test_average_cost_i = 0  # Initialize average cost of model i

            for batch in range(batch_epochs):
                start = batch * self.n_batch
                end = start + self.n_batch

                batch_data = test_data[start:end].T
                batch_labels = test_targets[start:end].T

                layers_out, class_out = self.forward(batch_data, testing=True)
                model_out[start:end] = class_out  # Add the batch predictions to the overall predictions

                loss, _ = self.cross_entropy_loss(layers_out[-1], batch_labels)
                test_average_loss_i += loss
                test_average_cost_i += loss + self.reg()

            models_loss[i] = test_average_loss_i / batch_epochs
            models_cost[i] = test_average_cost_i / batch_epochs
            models_accuracy[i] = self.accuracy(model_out, test_labels)  # Calculate the accuracy of each classifier
            models_out[i] = model_out  # Save the output of the model

        # Concatenate all results to a list
        results = []
        for i, model_results in models_out.items():
            results.append(model_results)
            # print("Model {} had {}% Test accuracy".format(i, models_accuracy[i] * 100))

        # Take majority vote across models
        average_out = mode(results, axis=0)[0]

        # Average accuracy over all models
        test_average_acc = self.accuracy(average_out, test_labels)
        # Average loss over all models
        test_average_loss = np.sum(list(models_loss.values())) / len(models_loss)
        # Average cost over all models
        test_average_cost = np.sum(list(models_cost.values())) / len(models_cost)

        return test_average_loss, test_average_cost, test_average_acc

    def forward(self, data, testing=False):
        """
        A forward pass in the network computing the predicted class
        """
        layers_out = []
        input_of_layer = data

        for layer in range(len(self.w) - 1):
            # calculate the ith hidden layer
            s_i = np.dot(self.w[layer], input_of_layer) + self.b[layer]

            if self.batch_norm is not None:
                # Normalize (scale & shift) output to a better distribution
                s_i = self.batch_norm.forward_per_layer(s_i, layer, testing=testing)

            # apply ReLU activation function
            h_i = self.relu(s_i)
            # save the output of that layer
            layers_out.append(h_i)
            # set the output of this hidden layer to be the input of the next
            input_of_layer = h_i

        # calculate the output layer
        s_out = np.dot(self.w[-1], input_of_layer) + self.b[-1]
        # apply softmax activation function
        p = self.softmax(s_out)
        # save the output of the output layer
        layers_out.append(p)
        # predicted class is label with highest probability
        k = np.argmax(p, axis=0)
        return layers_out, k

    def backward(self, l_out, data, targets):
        """
        A backward pass in the network to update the weights with gradient descent
        l_out: the output of each layer
        """
        # Compute the loss and its gradient using the network predictions and the real targets
        loss, loss_out_grad = self.cross_entropy_loss(l_out[-1], targets)

        # Add the L2 Regularization term (lambda * ||W||^2) to the loss
        cost = loss + self.reg()

        # Initialize list to save the gradients
        weights_grads = [None] * len(l_out)
        bias_grads = [None] * len(l_out)

        # Calculate output layer
        w_out_grad = np.dot(loss_out_grad, l_out[-2].T) / self.n_batch  # TODO: check if -1 or -2 (probably -2)
        b_out_grad = np.sum(loss_out_grad, axis=0) / self.n_batch
        reg_out_grad = 2 * self.lambda_reg * self.w[-1]
        weights_grads[-1] = w_out_grad + reg_out_grad
        bias_grads[-1] = b_out_grad

        loss_i_grad = np.dot(self.w[-1].T, loss_out_grad)  # Current (Next) layer's weights x current gradient
        indicator = l_out[-2] > 0  # indicator based on output previous layer output
        loss_i_grad = loss_i_grad * indicator

        # Update backwards, from last HIDDEN layer to SECOND HIDDEN layer. The first layer is dependant on the data
        for layer_i in range(len(l_out)-2, 0, -1):

            if self.batch_norm is not None:
                loss_i_grad = self.batch_norm.backward_per_layer(loss_i_grad, layer_i)

            # Calculate layer weight gradient based on the loss of that layer and the input of that layer
            w_i_grad = np.dot(loss_i_grad, l_out[layer_i-1].T) / self.n_batch
            # Calculate layer bias gradient based on its loss (maybe np.sum(loss_i_grad, axis=0) also works)
            b_i_grad = np.dot(loss_i_grad, np.ones((self.n_batch, 1))) / self.n_batch
            # Compute gradient of regularization term w.r.t the OUTPUT weights
            reg_i_grad = 2 * self.lambda_reg * self.w[layer_i]
            # Save the gradients
            weights_grads[layer_i] = w_i_grad + reg_i_grad
            bias_grads[layer_i] = b_i_grad

            # Calculate loss gradient of previous layer
            loss_i_grad = np.dot(self.w[layer_i].T, loss_i_grad)  # Current (Next) layer's weights x current gradient
            indicator = l_out[layer_i-1] > 0  # indicator based on output previous layer output
            loss_i_grad = loss_i_grad * indicator

        # Calculate FIRST hidden layer weight and bias gradients
        if self.batch_norm is not None:
            loss_i_grad = self.batch_norm.backward_per_layer(loss_i_grad, 0)

        w_0_grad = np.dot(loss_i_grad, data.T) / self.n_batch
        # Calculate layer bias gradient based on its loss
        b_0_grad = np.dot(loss_i_grad, np.ones((self.n_batch, 1))) / self.n_batch
        # Compute gradient of regularization term
        reg_0_grad = 2 * self.lambda_reg * self.w[0]
        # Save the gradients
        weights_grads[0] = w_0_grad + reg_0_grad
        bias_grads[0] = b_0_grad

        # Update backwards
        for i in range(len(weights_grads)-1, -1, -1):
            self.w[i] = self.w[i] - self.eta * weights_grads[i]
            self.b[i] = self.b[i] - self.eta * bias_grads[i]

        return loss, cost, weights_grads, bias_grads

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
        using the weights of ALL the layers
        """
        weight_sum = 0
        for w in self.w:
            weight_sum += np.sum(np.square(w))
        return self.lambda_reg * weight_sum

    def apply_noise(self, batch):
        """
        Add small amount of geometric noise to the training images to force the model
        to learn a more general representation of the data
        :return: a noisy batch
        """
        return batch + np.random.normal(self.noise_m, self.noise_std, batch.shape)

    def cycle_eta(self, iteration):
        """
        Calculate the learning rate for a specific cycle
        """
        cycle = iteration % (self.n_s * 2)
        diff = self.eta_max - self.eta_min

        if cycle < self.n_s:
            frac = cycle / self.n_s
            new_eta = self.eta_min + frac * diff
        else:  # or "when cycle < 2 * self.n_s"
            frac = (cycle - self.n_s) / self.n_s
            new_eta = self.eta_max - frac * diff

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

    def plot_train_val_progress(self):
        """
        Plot loss, cost, accuracy
        """
        self.general_plot(self.loss_train_av_history, self.loss_val_av_history, title="Loss", xlabel="Update steps")
        self.general_plot(self.cost_train_av_history, self.cost_val_av_history, title="Cost", xlabel="Update steps")
        self.general_plot(self.acc_train_av_history, self.acc_val_av_history, title="Accuracy", xlabel="Update steps")

    def general_plot(self, var_train, var_val, title=None, xlabel=None):
        """
        Plot the history of a variable
        """
        x_axis = range(1, len(var_train) + 1)
        # x_axis = np.arange(1, len(var_train) * self.n_batch + 1, self.n_batch)
        y_axis_train = var_train
        y_axis_val = var_val
        plt.plot(x_axis, y_axis_train, color='green', alpha=0.7, label="Train " + title)
        plt.plot(x_axis, y_axis_val, color='red', alpha=0.7, label="Validation " + title)
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(title)
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

    def plot_eta_history(self):
        """
        Plot the history of the error
        """
        x_axis = np.arange(1, len(self.eta_history) + 1)
        y_axis_eta = self.eta_history
        plt.plot(x_axis, y_axis_eta, alpha=0.7)
        plt.xlabel('Update steps')
        plt.ylabel('Eta values')
        plt.show()

    def compare_grads(self, network_structure, data, labels):
        """
        Compare the results of the analytical and the numerical (centered difference) calculations of the gradients
        """

        d = data.shape[1]  # number of features

        data = data.T
        labels = labels.T

        self.init_weights(network_structure, d)

        init_w = copy.deepcopy(self.w)
        init_b = copy.deepcopy(self.b)

        # Calculate numerically
        grad_w_num, grad_b_num, grad_gamma_num, grad_beta_num = self.compute_grads_num(data, labels)

        # Reset weights and bias to start from the same position
        self.w = copy.deepcopy(init_w)
        self.b = copy.deepcopy(init_b)

        # Calculate analytically
        l_out, _ = self.forward(data)
        _, _,  grad_w_ana, grad_b_ana = self.backward(l_out, data, labels)

        for i in range(len(grad_w_ana) - 1):
            layer = np.random.randint(0, len(grad_w_ana) - 1)
            node = np.random.randint(0, len(grad_w_ana[layer]))
            print("Random samples of hidden layer {} of neuron {}".format(layer, node))
            print(grad_w_num[layer][node][0:5])
            print(grad_w_ana[layer][node][0:5])

        print("Random samples of output layer of neuron 1")
        print(grad_w_num[-1][0][:5])
        print(grad_w_ana[-1][0][:5])

        print("Average error differences between the numerical and the analytical computation of gradients: ")
        for i in range(len(grad_w_ana) - 1):
            print("Hidden layer {} weights: {}".format(i, np.mean(np.abs(grad_w_ana[i]) - np.abs(grad_w_num[i]))))
            print("Hidden bias {} weights: {}".format(i, np.mean(np.abs(grad_b_ana[i]) - np.abs(grad_b_num[i]))))

        print("Output layer weights:", np.mean(np.abs(grad_w_ana[-1]) - np.abs(grad_w_num[-1])))
        print("Output layer bias:", np.mean(np.abs(grad_b_ana[-1]) - np.abs(grad_b_num[-1])))

    def compute_grads_num(self, data, targets):
        """
        Compute the gradients numerically to check if the analytic solution is correct
        """

        h = 1e-5

        grad_w = []
        grad_b = []

        grad_gamma = []
        grad_beta = []

        for j in range(len(self.b)):

            grad_b_j = np.zeros(len(self.b[j]))
            for i in range(len(self.b[j])):

                self.b[j][i] = self.b[j][i] - h

                layers_out, _ = self.forward(data)
                c1, _ = self.cross_entropy_loss(layers_out[-1], targets)
                c1 += self.reg()

                self.b[j][i] = self.b[j][i] + 2*h
                layers_out, _ = self.forward(data)
                c2, _ = self.cross_entropy_loss(layers_out[-1], targets)
                c2 += self.reg()

                self.b[j][i] = self.b[j][i] - h
                grad_b_j[i] = (c2 - c1) / (2 * h)

            grad_b.append(grad_b_j)

        for k in range(len(self.w)):

            grad_w_k = np.zeros(self.w[k].shape)

            for j in range(grad_w_k.shape[0]):

                for i in range(grad_w_k.shape[1]):
                    self.w[k][j][i] = self.w[k][j][i] - h

                    layers_out, _ = self.forward(data)
                    c1, _ = self.cross_entropy_loss(layers_out[-1], targets)
                    c1 += self.reg()

                    self.w[k][j][i] = self.w[k][j][i] + 2*h

                    layers_out, _ = self.forward(data)
                    c2, _ = self.cross_entropy_loss(layers_out[-1], targets)
                    c2 += self.reg()

                    self.w[k][j][i] = self.w[k][j][i] - h
                    grad_w_k[j][i] = (c2 - c1) / (2 * h)

            grad_w.append(grad_w_k)

            # last layer doesnt have gamma and beta
            if self.batch_norm is not None and k != len(self.w) - 1:
                grad_gamma_i, grad_beta_i = self.compute_batch_norm_grads_num(data, targets, k)
                grad_gamma.append(grad_gamma_i)
                grad_beta.append(grad_beta_i)

        return grad_w, grad_b, grad_gamma, grad_beta

    def compute_batch_norm_grads_num(self, data, targets, layer):

        h = 1e-5

        gamma_grad_i = np.zeros(self.batch_norm.gamma[layer].shape)

        for i in range(len(self.batch_norm.gamma[layer])):
            self.batch_norm.gamma[layer][i] = self.batch_norm.gamma[layer][i] - h

            layers_out, _ = self.forward(data)
            c1, _ = self.cross_entropy_loss(layers_out[-1], targets)
            c1 += self.reg()

            self.batch_norm.gamma[layer][i] = self.batch_norm.gamma[layer][i] + 2 * h

            layers_out, _ = self.forward(data)
            c2, _ = self.cross_entropy_loss(layers_out[-1], targets)
            c2 += self.reg()

            self.batch_norm.gamma[layer][i] = self.batch_norm.gamma[layer][i] - h
            gamma_grad_i[i] = (c2 - c1) / (2 * h)

        beta_grad_i = np.zeros(self.batch_norm.gamma[layer].shape)

        for i in range(len(self.batch_norm.beta[layer])):
            self.batch_norm.beta[layer][i] = self.batch_norm.beta[layer][i] - h

            layers_out, _ = self.forward(data)
            c1, _ = self.cross_entropy_loss(layers_out[-1], targets)
            c1 += self.reg()

            self.batch_norm.beta[layer][i] = self.batch_norm.beta[layer][i] + 2 * h

            layers_out, _ = self.forward(data)
            c2, _ = self.cross_entropy_loss(layers_out[-1], targets)
            c2 += self.reg()

            self.batch_norm.beta[layer][i] = self.batch_norm.beta[layer][i] - h
            beta_grad_i[i] = (c2 - c1) / (2 * h)

        return gamma_grad_i, beta_grad_i
