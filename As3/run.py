"""
Created by Kostis S-Z @ 2019-03-27
"""


import numpy as np
import os
import json
import datetime
from pathlib import Path
from network3 import MultiLayerNetwork
from data import load_data, preprocess_data, process_zero_mean
from matplotlib import pyplot as plt


parent_dir = str(Path.cwd().parent)  # Get the parent directory of the current working directory
directory = parent_dir + "/cifar-10-batches-py"  # The dataset should be in the parent directory


model_parameters = {
    # Basic variables
    "eta_min": 1e-5,  # min learning rate for cycle
    "eta_max": 1e-1,  # max learning rate for cycle
    "bn_cyc_eta": True,  # Use cyclical learning rate for BN
    "n_s": 500,  # parameter variable for cyclical learning rate
    "n_batch": 100,  # size of data batches within an epoch
    "init_type": "Xavier",  # Choose between Xavier and He initialisation
    "lambda_reg": 0.005,  # regularizing term variable
    # Extra variables
    "train_noisy": False,  # variable to toggle adding noise to the training data
    "noise_m": 0,  # the mean of the gaussian noise added to the training data
    "noise_std": 0.01,  # the standard deviation of the gaussian noise added to the training data
    "dropout": False,  # Use dropout or not
    "dropout_perc": 0.2,  # Percentage of nodes to dropout
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
}


# Train a network values
n_cycles = 2
model_parameters["n_s"] = (5 * 45000) / model_parameters["n_batch"]
epochs = int(2 * n_cycles * (model_parameters["n_s"] / model_parameters["n_batch"]))  # 48

save = True

# TODO: Batch Norm


# TODO: Lambda search
test_lambda = False
fine = True
# network_structure = layer3


# TODO: Sensitivity to initiliasation
# model_parameters["init_type"] = "He"
# net_sig = 1e-1  # 1e-3  1e-4
# model_parameters["n_s"] = (2 * 45000) / model_parameters["n_batch"]
# epochs = int(2 * n_cycles * (model_parameters["n_s"] / model_parameters["n_batch"]))  # 48

use_batch_norm = False
early_stop = False
ensemble = False

# index_of_layer : number_of_nodes
layer2 = {0: 50, 1: 10}
layer3 = {0: 50, 1: 50, 2: 10}
layer4 = {0: 50, 1: 50, 2: 10, 3: 10}
layer9 = {0: 50, 1: 30, 2: 20, 3: 20, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10}

network_structure = layer3


def main():
    # Use the loading function from Assignment 1
    train_x, train_y, val_x, val_y, test_x, test_y = load_data(use_all=True, val_size=10000)

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
    # train_simple(train_x, train_y, val_x, val_y, test_x, test_y)

    if test_lambda:
        lambda_search(train_x, train_y, val_x, val_y, test_x, test_y)
    else:
        train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)


def test_grad_computations(train_x, train_y):
    """
    Run one epoch and test if gradients are computed correctly
    """
    num_samples = 2
    num_features = 10

    train_x = train_x[:num_samples, :num_features]
    train_y = train_y[:num_samples]

    model_parameters["n_batch"] = num_samples  # size of data batches within an epoch
    model_parameters["eta"] = 0.01
    model_parameters["lambda_reg"] = 0.0

    net = MultiLayerNetwork(**model_parameters)

    net.compare_grads(network_structure, train_x, train_y, use_batch_norm=True)


def train_simple(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train a 2/3/9-layer network
    """
    n_s = 2 * int(train_x.shape[0] / model_parameters["n_batch"])
    model_parameters["n_s"] = n_s
    model_parameters["lambda_reg"] = 0.00087

    pa = 5
    cycles = 2
    n_s = (pa * 45000) / model_parameters["n_batch"]
    epochs = cycles * pa * 2
    model_parameters["lambda_reg"] = 0.005
    model_parameters["n_s"] = n_s

    net = MultiLayerNetwork(**model_parameters)

    net.train(network_structure, train_x, train_y, val_x, val_y,
              n_epochs=epochs, use_batch_norm=True, early_stop=False, ensemble=False, verbose=True)

    net.plot_train_val_progress()
    net.plot_eta_history()

    test_loss, test_cost, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a two-layer network
    """
    net = MultiLayerNetwork(**model_parameters)

    net.train(network_structure, train_x, train_y, val_data=val_x, val_labels=val_y, n_epochs=epochs,
              use_batch_norm=use_batch_norm, early_stop=early_stop, ensemble=ensemble, verbose=True)
    # net.train(network_structure, train_x, train_y, val_data=val_x, val_labels=val_y, n_epochs=epochs,
    #           use_batch_norm=True, early_stop=False, ensemble=False, verbose=True)

    test_loss, test_cost, test_accuracy = net.test(test_x, test_y)

    if save:
        model_id = save_model(test_accuracy)
    else:
        model_id = None

    net.plot_train_val_progress(save_dir=model_id)
    net.plot_eta_history()

    print("Test accuracy: ", test_accuracy * 100)

    return test_accuracy


def save_model(accuracy):
    now = datetime.datetime.now()
    model_id = str(now.day) + "_" + str(now.month) + "_" + str(now.hour) + "." + str(now.minute) + "/"

    os.makedirs(model_id)
    with open(model_id + 'model_params.txt', 'w') as f:
        f.write("Results: \n")
        f.write("  Accuracy: " + str(100 * accuracy) + "%\n\n")

        f.write("Model parameters: \n")
        f.write("  Epochs : " + str(epochs) + "\n")
        f.write("  Batch Normalization : " + str(use_batch_norm) + "\n")
        f.write("  Early Stopping : " + str(early_stop) + "\n")
        f.write("  Ensemble : " + str(ensemble) + "\n")

        for key, value in model_parameters.items():
            f.write("  " + key + " : " + str(value) + "\n")
        f.write("\nNetwork architecture: \n")
        for key, value in network_structure.items():
            f.write("  " + str(key) + " : " + str(value) + "\n")

    return model_id


def lambda_search(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Search for the optimal lambda
    """
    # Coarse search
    l_min = -5
    l_max = -1
    n_lambda = 20
    # Fine search
    l_min_f = -4
    l_max_f = -2
    n_lambda_f = 20

    if fine:
        l_min = l_min_f
        l_max = l_max_f
        n_lambda = n_lambda_f

    lambda_regs = []
    for _ in range(n_lambda):
        l_i = l_min + (l_max - l_min) * np.random.rand()
        lambda_regs.append(10 ** l_i)

    results = []
    optimal_lambda = 0
    best_model_accuracy = 0.

    for lambda_reg in lambda_regs:
        model_parameters["lambda_reg"] = lambda_reg

        test_acc = train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)

        test_acc = round(test_acc * 100, 1)
        print("Lambda: {} | Test accuracy: {}".format(lambda_reg, test_acc))

        results.append((lambda_reg, test_acc))

        if test_acc > best_model_accuracy:
            best_model_accuracy = test_acc
            optimal_lambda = lambda_reg

    print("Optimal lambda: {} with test accuracy: {}".format(optimal_lambda, best_model_accuracy))

    if fine:
        title = "fine"
    else:
        title = "coarse"

    results_dict = dict((k, v) for k, v in results)
    with open('lambda_results_' + title + '.json', 'w') as fp:
        json.dump(results_dict, fp, sort_keys=True, indent=2)

    x_axis = [x[0] for x in results]
    y_axis = [x[1] for x in results]
    plt.plot(x_axis, y_axis, color='red', alpha=0.6, linestyle='-', marker='o')
    plt.xlabel("Lambda values")
    plt.ylabel("Test Accuracy")
    plt.show()


if __name__ == "__main__":
    main()
