"""
THIS FILE IS IDENTICAL IN MOST PARTS WITH THE RUN FILE FOUND IN ASSIGNMENT 1

Created by Kostis S-Z @ 2019-03-27
"""


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
    1: 10,
}

#     2: 50,
#     3: 10,  # Output layer should have the same number of nodes as classes to predict
# }


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
    net = MultiLayerNetwork(**model_parameters)

    model_parameters["n_s"] = (5 * 45000) / model_parameters["n_batch"]

    net.train(network_structure, train_x, train_y, val_x, val_y,
              n_epochs=20, early_stop=False, ensemble=False, verbose=True)

    net.plot_loss()  # Plot the loss progress
    net.plot_eta_history()

    test_loss, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


if __name__ == "__main__":
    main()
