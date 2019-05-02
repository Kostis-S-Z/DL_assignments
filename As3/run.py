"""
Created by Kostis S-Z @ 2019-03-27
"""


import numpy as np
from pathlib import Path
from network3 import MultiLayerNetwork
from data import load_data, preprocess_data, process_zero_mean

parent_dir = str(Path.cwd().parent)  # Get the parent directory of the current working directory
directory = parent_dir + "/cifar-10-batches-py"  # The dataset should be in the parent directory


model_parameters = {
    "eta_min": 1e-5,  # min learning rate for cycle
    "eta_max": 1e-1,  # max learning rate for cycle
    "n_s": 500,  # parameter variable for cyclical learning rate
    "n_batch": 100,  # size of data batches within an epoch
    "init_type": "Xavier",  # Choose between Xavier and He initialisation
    "dropout": False,  # Use dropout or not
    "dropout_perc": 0.2,  # Percentage of nodes to dropout
    "train_noisy": False,  # variable to toggle adding noise to the training data
    "noise_m": 0,  # the mean of the gaussian noise added to the training data
    "noise_std": 0.01,  # the standard deviation of the gaussian noise added to the training data
    "lambda_reg": 0.005,  # regularizing term variable
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
}

# index_of_layer : number_of_nodes
network_structure = {
    0: 50,
    1: 50,
    2: 50,
    3: 10  # Output layer should have the same number of nodes as classes to predict
}


def main():
    # Use the loading function from Assignment 1
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(use_all=True, val_size=5000)

    # Use the preprocessing function from Assignment 1
    train_x, train_y = preprocess_data(train_x, train_y)
    val_x, val_y = preprocess_data(val_x, val_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    # Process the data so they have a zero mean
    mean, std = np.mean(train_x), np.std(train_x)  # Find mean and std of training data
    train_x = process_zero_mean(train_x, mean, std)
    val_x = process_zero_mean(val_x, mean, std)
    test_x = process_zero_mean(test_x, mean, std)

    # Testing gradients
    # test_grad_computations(train_x, train_y)

    # Training simple k-layer networks
    train_simple(train_x, train_y, val_x, val_y, test_x, test_y)

    # train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)


def test_grad_computations(train_x, train_y):
    """
    Run one epoch and test if gradients are computed correctly
    """
    num_samples = 2
    num_features = 30

    train_x = train_x[:num_samples, :num_features]
    train_y = train_y[:num_samples]

    model_parameters["n_batch"] = num_samples  # size of data batches within an epoch
    model_parameters["eta"] = 0.01
    model_parameters["lambda_reg"] = 0.0

    net = MultiLayerNetwork(**model_parameters)

    net.compare_grads(network_structure, train_x, train_y)


def train_simple(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train a 2/3/9-layer network
    """
    n_s = 2 * int(train_x.shape[0] / model_parameters["n_batch"])
    model_parameters["n_s"] = n_s
    model_parameters["lambda_reg"] = 0.00087
    layer2 = {0: 50, 1: 10}

    pa = 5
    cycles = 2
    n_s = (pa * 45000) / model_parameters["n_batch"]
    epochs = cycles * pa * 2
    model_parameters["lambda_reg"] = 0.005
    model_parameters["n_s"] = n_s

    layer3 = {0: 50, 1: 50, 2: 10}
    layer9 = {0: 50, 1: 30, 2: 20, 3: 20, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10}

    network_structure = layer3

    net = MultiLayerNetwork(**model_parameters)

    net.train(network_structure, train_x, train_y, val_x, val_y,
              n_epochs=epochs, batch_norm=False, early_stop=False, ensemble=False, verbose=True)

    net.plot_train_val_progress()
    net.plot_eta_history()

    test_loss, test_cost, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a two-layer network
    """
    net = MultiLayerNetwork(**model_parameters)
    epochs = 20

    model_parameters["n_s"] = (5 * 45000) / model_parameters["n_batch"]

    net.train(network_structure, train_x, train_y, val_x, val_y,
              n_epochs=epochs, early_stop=False, ensemble=False, verbose=True)

    net.plot_train_val_progress()
    net.plot_eta_history()

    test_loss, test_cost, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
