"""
THIS FILE IS IDENTICAL IN MOST PARTS WITH THE RUN FILE FOUND IN ASSIGNMENT 1

Created by Kostis S-Z @ 2019-03-27
"""

import numpy as np
from pathlib import Path
from As1 import run

parent_dir = str(Path.cwd().parent)  # Get the parent directory of the current working directory
directory = parent_dir + "/cifar-10-batches-py"  # The dataset should be in the parent directory


model_parameters = {
    "eta": 0.01,  # learning rate
    "n_batch": 100,  # size of data batches within an epoch
    "n_nodes": 50,  # number of nodes (neurons) in the hidden layer
    "loss_type": "cross-entropy",  # cross-entropy or svm
    "lambda_reg": 0.,  # regularizing term variable
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 10  # how many epochs to wait before stopping training if the val_error is below min_delta
}


def main():
    # Use the loading function from Assignment 1
    train_x, train_y, val_x, val_y, test_x, test_y = run.load_data()

    # Use the preprocessing function from Assignment 1
    train_x, train_y = run.preprocess_data(train_x, train_y)
    val_x, val_y = run.preprocess_data(val_x, val_y)
    test_x, test_y = run.preprocess_data(test_x, test_y)

    # Process the data so they have a zero mean
    train_x, train_y = process_zero_mean(train_x)
    val_x, val_y = process_zero_mean(val_x)
    test_x, test_y = process_zero_mean(test_x)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a multi-layer network
    """


def process_zero_mean(data):
    """
    Preprocess data to have a zero mean
    :return: the processed data
    """
    return data


if __name__ == "__main__":
    main()
