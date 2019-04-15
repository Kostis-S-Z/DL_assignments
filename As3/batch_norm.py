import numpy as np


def batch_norm(self, data):
    """
    Scale and shift the (unfiltered) output of a layer
    """
    layers_out = []
    unfilt_out = []  # this will be size layers - 1, since we dont add the output
    bn_out = []  # this will be size layers - 1, since we dont add the output
    means = []
    vars = []
    betas = []
    gammas = []
    input_of_layer = data
    for layer in range(len(self.w) - 1):
        # calculate the ith hidden layer
        s_i = np.dot(self.w[layer], input_of_layer) + self.b[layer]
        unfilt_out.append(s_i)
        mean = np.mean(s_i)
        var = np.std(s_i)
        s_hat_i = (s_i - mean) / var
        means.append(mean)
        vars.append(var)
        bn_out.append(s_hat_i)
        gamma = 0
        beta = 0
        s_til_i = gamma * s_hat_i + beta
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

    self.moving_av()
    return layers_out, k


def backward(self, l_out, unfilt_out, bn_out, means, vars, betas, gammas, data, targets):
    """
    A backward pass in the network to update the weights with gradient descent
    l_out: the output of each layer
    """
    # Compute the loss and its gradient using the network predictions and the real targets
    loss, loss_out_grad = self.cross_entropy_loss(l_out[-1], targets)

    # Add the L2 Regularization term (lambda * ||W||^2) to the loss
    loss = loss + self.reg()

    # Set the loss gradient of the output layer to use it for the update
    # Initialize list to save the gradients
    weights_grads = [None] * len(l_out)
    bias_grads = [None] * len(l_out)

    gammas_grads = [None] * len(l_out-1)
    betas_grads = [None] * len(l_out-1)

    # Calculate output layer
    w_out_grad = np.dot(loss_out_grad, l_out[-2].T) / self.n_batch  # TODO: check if -1 or -2 (probably -2)
    b_out_grad = np.sum(loss_out_grad, axis=0) / self.n_batch
    reg_out_grad = 2 * self.lambda_reg * self.w[-1]
    weights_grads[-1] = w_out_grad + reg_out_grad
    bias_grads[-1] = b_out_grad

    loss_i_grad = np.dot(self.w[-1].T, loss_out_grad)  # Current (Next) layer's weights x current gradient
    indicator = l_out[-2] > 0  # indicator based on output previous layer output
    loss_i_grad = loss_i_grad * indicator

    # Update backwards, from ONE BEFORE OUTPUT layer to SECOND layer. The first layer is dependent on the data
    for layer_i in range(len(l_out)-2, 0, -1):

        o_grad = loss_i_grad * bn_out[layer_i]
        gamma_i_grad = np.dot(o_grad, np.ones((self.n_batch, 1))) / self.n_batch
        beta_i_grad = np.dot(loss_i_grad, np.ones((self.n_batch, 1))) / self.n_batch

        gammas_grads[layer_i] = gamma_i_grad
        betas_grads[layer_i] = beta_i_grad

        g_out = np.dot(gammas[layer_i], np.ones((self.n_batch, 1)).T)
        loss_i_grad = loss_i_grad * g_out

        loss_i_grad = self.bn_backpass(loss_i_grad, unfilt_out[layer_i], means[layer_i], vars[layer_i])

        w_i_grad = np.dot(loss_i_grad, l_out[layer_i-1].T) / self.n_batch
        b_i_grad = np.sum(loss_i_grad, axis=0) / self.n_batch

        reg_i_grad = 2 * self.lambda_reg * self.w[layer_i-1]

        weights_grads[layer_i] = w_i_grad + reg_i_grad
        bias_grads[layer_i] = b_i_grad

        loss_i_grad = np.dot(self.w[layer_i].T, loss_i_grad)
        indicator = l_out[layer_i - 1] > 0
        loss_i_grad = loss_i_grad * indicator

    # Calculate FIRST hidden layer weight and bias gradients
    o_grad = loss_i_grad * bn_out[0]
    gamma_0_grad = np.dot(o_grad, np.ones((self.n_batch, 1))) / self.n_batch
    beta_0_grad = np.dot(loss_i_grad, np.ones((self.n_batch, 1))) / self.n_batch

    gammas_grads[0] = gamma_0_grad
    betas_grads[0] = beta_0_grad

    g_out = np.dot(gammas[0], np.ones((self.n_batch, 1)).T)
    loss_0_grad = loss_i_grad * g_out

    loss_0_grad = self.bn_backpass(loss_0_grad, unfilt_out[0], means[0], vars[0])

    w_0_grad = np.dot(loss_0_grad, data.T) / self.n_batch
    # Calculate layer bias gradient based on its loss
    b_0_grad = np.sum(loss_0_grad, axis=0) / self.n_batch
    # Compute gradient of regularization term
    reg_0_grad = 2 * self.lambda_reg * self.w[0]
    # Save the gradients
    weights_grads[0] = w_0_grad + reg_0_grad
    bias_grads[0] = b_0_grad

    # Update backwards
    for i in range(len(weights_grads)-1, -1, -1):
        self.w[i] = self.w[i] - self.eta * weights_grads[i]
        self.b[i] = self.b[i] - self.eta * bias_grads[i]

    return loss


def bn_backpass(self, g_batch, unfilt_out, mean_i, var_i):
    """

    """
    ones = np.ones((self.n_batch, 1))

    epsilon = 0.01
    sigma_1 = ( var_i + epsilon ) ** (-0.5)
    sigma_1 = sigma_1.T

    sigma_2 = ( var_i + epsilon ) ** (-1.5)
    sigma_2 = sigma_2.T

    p1 = np.dot(sigma_1, ones.T)
    g1 = g_batch * p1

    p2 = np.dot(sigma_2, ones.T)
    g2 = g_batch * p2

    pm = np.dot(mean_i, ones.T)
    d = unfilt_out - pm

    c = g2 * d
    c = np.dot(c, ones)

    part1 = np.dot(g1, ones) / self.n_batch
    part2a = np.dot(c, ones.T)
    part2 = (d * part2a) / self.n_batch
    new_g_batch = g1 - part1 - part2
    return new_g_batch


def moving_av(self, mean_i, var_i):
    alpha = 0.9  # smaller than 0.9
    m_av_i = 0  # init
    var_av_i = 0  # init
    m_av_i = alpha * m_av_i + (1 - alpha) * mean_i
    var_av_i = alpha * var_av_i + (1 - alpha) * var_i
