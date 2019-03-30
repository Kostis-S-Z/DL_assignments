"""
Created by Kostis S-Z @ 2019-03-27
"""

import numpy as np
import matplotlib as plt
from network import OneLayerNetwork


directory = "cifar-10-batches-py"

model_parameters = {
    "eta": 0.01,  # learning rate
    "momentum": 0.,  # momentum
    "n_batch": 100,  # size of data batches within an epoch
    "lambda_reg": 0.,  # regularizing term variable
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 10  # how many epochs to wait before stopping training if the val_error is below min_delta
}


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = load_data()

    train_x, train_y = preprocess_data(train_x, train_y)
    val_x, val_y = preprocess_data(val_x, val_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    # train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)

    grid_search(train_x, train_y, val_x, val_y, test_x, test_y)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a single network
    """
    net = OneLayerNetwork(**model_parameters)

    net.train(train_x, train_y, val_x, val_y, n_epochs=40, early_stop=False, verbose=True)

    test_error = net.test(test_x, test_y)

    print("Test error: ", test_error)


def grid_search(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Grid search to find the optimal values for the hyper parameters of the network
    """

    eta_s = [0.01, 0.05, 0.1, 0.15]
    lambda_reg_s = [0.001, 0.01, 0.05, 0.1]
    n_batch_s = [10, 50, 100, 500]

    results = {}
    optimal_parameters = []
    best_model_error = 1.

    for eta in eta_s:
        model_parameters["eta"] = eta

        for lambda_reg in lambda_reg_s:
            model_parameters["lambda_reg"] = lambda_reg

            for n_batch in n_batch_s:
                model_parameters["n_batch"] = n_batch

                net = OneLayerNetwork(**model_parameters)

                net.train(train_x, train_y, val_x, val_y, n_epochs=1, early_stop=False, verbose=True)

                test_error = net.test(test_x, test_y)

                key = [eta, lambda_reg, n_batch]

                results[str(key)] = test_error

                if test_error < best_model_error:
                    best_model_error = test_error
                    optimal_parameters = key

    print("Optimal model parameters:")
    print("     eta: ", optimal_parameters[0])
    print("     lambda_reg: ", optimal_parameters[1])
    print("     n_batch: ", optimal_parameters[2])
    print("  Test error: ", best_model_error)


def load_data():
    """
    for this assignment we use the data in the following way:
    training data: batch 1
    validation data: batch 2
    test data: test_batch
    source: http://www.cs.toronto.edu/~kriz/cifar.html
    data:
         a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
         The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
         The image is stored in row-major order, so that the first 32 entries of the array are the red channel values
         of the first row of the image.
    labels:
        a list of 10000 numbers in the range 0-9.
        The number at index i indicates the label of the ith image in the array data.
    :return: the data with their labels
    """
    import pickle

    train_file = directory + "/data_batch_1"

    with open(train_file, 'rb') as fo:
        train_data = pickle.load(fo, encoding='bytes')

    val_file = directory + "/data_batch_2"
    with open(val_file, 'rb') as fo:
        val_data = pickle.load(fo, encoding='bytes')

    test_file = directory + "/test_batch"
    with open(test_file, 'rb') as fo:
        test_data = pickle.load(fo, encoding='bytes')

    return train_data[b"data"], train_data[b"labels"], val_data[b"data"], \
        val_data[b"labels"], test_data[b"data"], test_data[b"labels"]


def preprocess_data(data, labels):
    """
    Preprocess data by normalizing between [0,1] and convert labels to one hot
    :return: the preprocessed data
    """
    data = data / 255
    labels = np.eye(10)[labels]

    return data, labels


def plot_loss(self):
    """
    Plot the history of the error
    """
    x_axis = range(1, len(self.error_history) + 1)
    y_axis = self.error_history
    plt.scatter(x_axis, y_axis, alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
