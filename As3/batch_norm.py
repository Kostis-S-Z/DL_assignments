import numpy as np

# Prevent division by zero
e = 1e-16


class BatchNormalization:
    """
    Class to save parameters of Batch Normalization method
    """

    def __init__(self, net_structure):

        self.gamma = []
        self.beta = []

        self.gamma_grads = []
        self.beta_grads = []

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

        self.gamma_grads = [None] * len(net_structure - 1)
        self.beta_grads = [None] * len(net_structure - 1)

    def forward_per_layer(self, s_i, layer):
        """
        Scale and shift
        :param s_i: unnormalized output of a layer
        :param layer: the index of the layer
        :return: the normalized output
        """
        self.l_out_unnorm.append(s_i)

        # Calculate mean and variance over the samples (the batch size)
        mean_i = np.mean(s_i, axis=0)
        var_i = np.var(s_i, axis=0)

        # Scale and shift to a normalized activation
        s_i_norm = (s_i - mean_i) / ((var_i + e) ** (-0.5))

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

        loss_i_grad = self.bn_backpass(loss_i_grad)

        # If you are in the first layer, update all gamma and beta from the gradients
        if layer_i == 0:
            # Update backwards
            for i in range(len(self.gamma) - 1, -1, -1):
                # TODO: how do you update gamma and beta exactly? do you use a learning rate?
                self.gamma[i] = self.gamma[i] - self.gamma_grads[i]
                self.beta[i] = self.beta[i] - self.beta_grads[i]

        return loss_i_grad

    def bn_backpass(self, g_batch, i):
        """

        """
        ones = np.ones((self.n_batch, 1))

        epsilon = 0.01
        sigma_1 = (self.layer_vars[i] + epsilon ) ** (-0.5)
        sigma_1 = sigma_1.T

        sigma_2 = ( self.layer_vars[i] + epsilon ) ** (-1.5)
        sigma_2 = sigma_2.T

        p1 = np.dot(sigma_1, ones.T)
        g1 = g_batch * p1

        p2 = np.dot(sigma_2, ones.T)
        g2 = g_batch * p2

        pm = np.dot(self.layer_means[i], ones.T)
        d = self.l_out_unnorm[i] - pm

        c = g2 * d
        c = np.dot(c, ones)

        part1 = np.dot(g1, ones) / self.n_batch
        part2a = np.dot(c, ones.T)
        part2 = (d * part2a) / self.n_batch
        new_g_batch = g1 - part1 - part2
        return new_g_batch

    def moving_av(self, i):
        alpha = 0.9  # smaller than 0.9
        m_av_i = 0  # init
        var_av_i = 0  # init
        m_av_i = alpha * m_av_i + (1 - alpha) * self.layer_means[i]
        var_av_i = alpha * var_av_i + (1 - alpha) * self.layer_vars[i]

        return m_av_i, var_av_i