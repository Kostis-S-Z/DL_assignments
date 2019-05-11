import numpy as np

# Prevent division by zero
e = 1e-16


class BatchNormalization:
    """
    Class to save parameters of Batch Normalization method

    means: mx1 (a mean per node)
    vars: mx1 (a variance per node)
    """

    def __init__(self, net_structure, batch_size):

        self.n_batch = batch_size
        self.alpha = 0.7  # smaller than 0.9

        self.gamma = []
        self.beta = []

        self.gamma_grads = []
        self.beta_grads = []

        self.m_av = []
        self.var_av = []

        self.layer_means = []
        self.layer_vars = []

        self.l_out_unnorm = []
        self.l_out_norm = []
        self.l_out_fit = []

        self.init_values(net_structure)

    def init_values(self, net_structure):
        mean = 0
        std = 0.1

        # We don't have gamma and beta for the output layer, that's why net_structure - 1!
        for l in range(len(net_structure) - 1):
            # Choose between zeros, ones, random initialization
            zeros = np.zeros((net_structure[l], 1))
            ones = np.ones((net_structure[l], 1))
            random = np.random.normal(mean, std, (net_structure[l], 1))
            # Initialize beta & gamma
            self.gamma.append(random)
            self.beta.append(zeros)
            # Initialize moving average of mean and variance
            self.m_av.append(0)
            self.var_av.append(0)

        self.gamma_grads = [None] * (len(net_structure) - 1)
        self.beta_grads = [None] * (len(net_structure) - 1)

    def forward_per_layer(self, s_i, layer, testing=False):
        """
        Scale and shift
        :param s_i: unnormalized output of a layer
        :param layer: the index of the layer
        :param testing: use precomputed mean and average during testing
        :return: the normalized output
        """
        self.l_out_unnorm.append(s_i)

        if testing:
            # This is used for testing
            mean_i = self.m_av[layer]
            var_i = self.var_av[layer]
        else:
            # Calculate mean and variance over the un-normalized samples (the batch size)
            mean_i = np.mean(s_i, axis=1)
            # both ways give the same result
            var_i = np.var(s_i, axis=1)  # maybe compensate with * (n-1) / n)
            # var_i = np.sum(((s_i.T - mean_i) ** 2 / self.n_batch), axis=0)

        # Scale and shift to a normalized activation
        s_i_norm = (s_i.T - mean_i) / ((var_i + e) ** (-0.5))
        s_i_norm = s_i_norm.T

        # Update s
        s_i = self.gamma[layer] * s_i_norm + self.beta[layer]

        # Save outputs
        self.layer_means.append(mean_i)
        self.layer_vars.append(var_i)
        self.l_out_norm.append(s_i_norm)
        self.l_out_fit.append(s_i)

        return s_i

    def backward_per_layer(self, loss_i_grad, layer_i):
        """
        A backward pass with Batch Normalization
        :return normalized loss gradient
        """

        o_grad = loss_i_grad * self.l_out_norm[layer_i]
        gamma_i_grad = np.dot(o_grad, np.ones((self.n_batch, 1))) / self.n_batch
        beta_i_grad = np.dot(loss_i_grad, np.ones((self.n_batch, 1))) / self.n_batch

        self.gamma_grads[layer_i] = gamma_i_grad
        self.beta_grads[layer_i] = beta_i_grad

        g_out = np.dot(self.gamma[layer_i], np.ones((self.n_batch, 1)).T)
        loss_i_grad = loss_i_grad * g_out

        loss_i_grad = self.bn_backpass(loss_i_grad, layer_i)

        # If you are in the first layer, update all gamma and beta from the gradients
        if layer_i == 0:
            # Update backwards
            for i in range(len(self.gamma) - 1, -1, -1):
                # TODO: how do you update gamma and beta exactly? do you use a learning rate?
                self.gamma[i] = self.gamma[i] - self.gamma_grads[i]
                self.beta[i] = self.beta[i] - self.beta_grads[i]
                self.update_moving_av(i)

        return loss_i_grad

    def bn_backpass(self, g_batch, i):
        ones = np.ones((self.n_batch, 1))

        sigma_1 = (self.layer_vars[i] + e) ** (-0.5)
        sigma_1 = sigma_1.T.reshape((-1, 1))

        sigma_2 = (self.layer_vars[i] + e) ** (-1.5)
        sigma_2 = sigma_2.T.reshape((-1, 1))

        p1 = np.dot(sigma_1, ones.T)
        g1 = g_batch * p1

        p2 = np.dot(sigma_2, ones.T)
        g2 = g_batch * p2

        mean_i = self.layer_means[i].reshape((-1, 1))
        pm = np.dot(mean_i, ones.T)
        d = self.l_out_unnorm[i] - pm

        c = g2 * d
        c = np.dot(c, ones)

        part1 = np.dot(g1, ones) / self.n_batch
        part2a = np.dot(c, ones.T)
        part2 = (d * part2a) / self.n_batch
        new_g_batch = g1 - part1 - part2
        return new_g_batch

    def update_moving_av(self, i):
        self.m_av[i] = self.alpha * self.m_av[i] + (1 - self.alpha) * self.layer_means[i]
        self.var_av[i] = self.alpha * self.var_av[i] + (1 - self.alpha) * self.layer_vars[i]
