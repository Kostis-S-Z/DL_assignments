"""
THIS FILE IS IDENTICAL IN MOST PARTS WITH THE RUN FILE FOUND IN ASSIGNMENT 1

Created by Kostis S-Z @ 2019-03-27
"""

from pathlib import Path
from network2 import TwoLayerNetwork
from data import load_data, preprocess_data, process_zero_mean

parent_dir = str(Path.cwd().parent)  # Get the parent directory of the current working directory
directory = parent_dir + "/cifar-10-batches-py"  # The dataset should be in the parent directory


model_parameters = {
    "eta_min": 0.001,  # min learning rate for cycle
    "eta_max": 0.1,  # max learning rate for cycle
    "n_s": 500,  # parameter variable for cyclical learning rate
    "n_batch": 200,  # size of data batches within an epoch
    "n_nodes": 50,  # number of nodes (neurons) in the hidden layer
    "train_noisy": False,  # variable to toggle adding noise to the training data
    "noise_m": 0,  # the mean of the gaussian noise added to the training data
    "noise_std": 0.01,  # the standard deviation of the gaussian noise added to the training data
    "lambda_reg": 0.,  # regularizing term variable
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
}


def main():
    # Use the loading function from Assignment 1
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(use_all=True)

    # Use the preprocessing function from Assignment 1
    train_x, train_y = preprocess_data(train_x, train_y)
    val_x, val_y = preprocess_data(val_x, val_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    # Process the data so they have a zero mean
    train_x, val_x, test_x = process_zero_mean(train_x, val_x, test_x)

    train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a two-layer network
    """
    net = TwoLayerNetwork(**model_parameters)

    net.train(train_x, train_y, val_x, val_y, n_epochs=40, early_stop=False, ensemble=True, verbose=True)

    # net.plot_loss()  # Plot the loss progress

    test_loss, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
