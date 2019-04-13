"""
THIS FILE IS IDENTICAL IN MOST PARTS WITH THE RUN FILE FOUND IN ASSIGNMENT 1

Created by Kostis S-Z @ 2019-03-27
"""

import numpy as np
from pathlib import Path
from network2 import TwoLayerNetwork
from data import load_data, preprocess_data, process_zero_mean

parent_dir = str(Path.cwd().parent)  # Get the parent directory of the current working directory
directory = parent_dir + "/cifar-10-batches-py"  # The dataset should be in the parent directory


model_parameters = {
    "eta_min": 1e-5,  # min learning rate for cycle
    "eta_max": 1e-1,  # max learning rate for cycle
    "n_s": 500,  # parameter variable for cyclical learning rate
    "n_batch": 100,  # size of data batches within an epoch
    "train_noisy": False,  # variable to toggle adding noise to the training data
    "noise_m": 0,  # the mean of the gaussian noise added to the training data
    "noise_std": 0.01,  # the standard deviation of the gaussian noise added to the training data
    "lambda_reg": 0.,  # regularizing term variable
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
}


# index_of_layer : number_of_nodes
network_structure = {
    0: 50,
    1: 10,  # Output layer should have the same number of nodes as classes to predict
}


def main():
    # Use the loading function from Assignment 1
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(use_all=True, val_size=5000)

    # Use the preprocessing function from Assignment 1
    train_x, train_y = preprocess_data(train_x, train_y)
    val_x, val_y = preprocess_data(val_x, val_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    # Process the data so they have a zero mean
    train_x, val_x, test_x = process_zero_mean(train_x, val_x, test_x)

    # test_grad_computations(train_x, train_y)

    # train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)

    # best_lambda = lambda_search(train_x, train_y, val_x, val_y)


def test_grad_computations(train_x, train_y):
    """
    Run one epoch and test if gradients are computed correctly
    """
    num_samples = 1
    num_features = 20

    train_x = train_x[:num_samples, :num_features]
    train_y = train_y[:num_samples]

    model_parameters["n_batch"] = num_samples  # size of data batches within an epoch
    model_parameters["eta"] = 0.01  # size of data batches within an epoch
    model_parameters["lambda_reg"] = 0.0  # size of data batches within an epoch

    net = TwoLayerNetwork(**model_parameters)

    net.compare_grads(network_structure, train_x, train_y)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a two-layer network
    """
    net = TwoLayerNetwork(**model_parameters)

    net.train(network_structure, train_x, train_y, val_x, val_y,
              n_epochs=40, early_stop=False, ensemble=True, verbose=True)

    net.plot_loss()  # Plot the loss progress
    net.plot_eta_history()

    test_loss, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


def lambda_search(train_x, train_y, val_x, val_y):
    """
    Search for the optimal lambda
    """

    lambda_reg_coarse = [0.001, 0.01, 0.05, 0.1]
    lambda_reg_medium = np.arange(0.0001, 0.2, 0.05)
    lambda_reg_fine = np.arange(0.0001, 0.2, 0.005)

    lambda_reg_s = lambda_reg_coarse

    results = {}
    optimal_lambda = 0
    best_model_accuracy = 0.

    epochs = 50
    model_parameters["n_s"] = 2 * int(epochs / model_parameters["n_batch"])

    for lambda_reg in lambda_reg_s:
        model_parameters["lambda_reg"] = lambda_reg

        net = TwoLayerNetwork(**model_parameters)

        val_accuracy = net.train(train_x, train_y, val_x, val_y, n_epochs=epochs, early_stop=False, verbose=False)

        val_acc = round(val_accuracy * 100, 1)
        print("Lambda: {} | Test accuracy: {}".format(lambda_reg, val_acc))

        results[lambda_reg] = val_accuracy

        if val_accuracy > best_model_accuracy:
            best_model_accuracy = val_accuracy
            optimal_lambda = lambda_reg

    print("Optimal lambda: {} with test accuracy: {}".format(optimal_lambda, best_model_accuracy))
    return optimal_lambda


if __name__ == "__main__":
    main()
