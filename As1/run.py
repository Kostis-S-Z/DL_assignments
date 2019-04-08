"""
Created by Kostis S-Z @ 2019-03-27
"""

from network import OneLayerNetwork
from data import load_data, preprocess_data


labels_to_names = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}

model_parameters = {
    "eta": 0.001,  # learning rate
    "n_batch": 100,  # size of data batches within an epoch
    "loss_type": "svm",  # cross-entropy or svm
    "svm_margin": 1.,  # margin parameter for svm loss
    "lambda_reg": .1,  # regularizing term variable
    "min_delta": 0.01,  # minimum accepted validation error
    "patience": 40  # how many epochs to wait before stopping training if the val_error is below min_delta
}


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = load_data()

    train_x, train_y = preprocess_data(train_x, train_y)
    val_x, val_y = preprocess_data(val_x, val_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    train_a_network(train_x, train_y, val_x, val_y, test_x, test_y)

    # grid_search(train_x, train_y, val_x, val_y, test_x, test_y)


def train_a_network(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Train and test a single network
    """
    net = OneLayerNetwork(**model_parameters)

    net.train(train_x, train_y, val_x, val_y, n_epochs=40, early_stop=True, verbose=True)

    net.plot_loss()  # Plot the loss progress

    # Plot the learnt representations of the weight matrices
    for i, name in labels_to_names.items():
        net.plot_weight_matrix(i, name)

    test_loss, test_accuracy = net.test(test_x, test_y)

    print("Test accuracy: ", test_accuracy)


def grid_search(train_x, train_y, val_x, val_y, test_x, test_y):
    """
    Grid search to find the optimal values for the hyper parameters of the network
    """

    eta_s = [0.005, 0.01, 0.05, 0.1]
    lambda_reg_s = [0.001, 0.01, 0.05, 0.1]
    n_batch_s = [10, 50, 100, 500]

    results = {}
    optimal_parameters = []
    best_model_accuracy = 0.

    print("Model parameters | Test accuracy")
    for eta in eta_s:
        model_parameters["eta"] = eta

        for lambda_reg in lambda_reg_s:
            model_parameters["lambda_reg"] = lambda_reg

            for n_batch in n_batch_s:
                model_parameters["n_batch"] = n_batch
                print("Initializing Network with:")
                print("       eta: {} lambda: {} batch_size: {}".format(eta, lambda_reg, n_batch))

                net = OneLayerNetwork(**model_parameters)

                net.train(train_x, train_y, val_x, val_y, n_epochs=50, early_stop=False, verbose=False)

                test_loss, test_accuracy = net.test(test_x, test_y)
                test_acc = round(test_accuracy * 100, 1)
                # print("eta: {} lambda: {} batch_size: {}  | test accuracy: {}%".format(eta, lambda_reg, n_batch, test_acc))
                print("     Test accuracy: ", test_acc)

                key = [eta, lambda_reg, n_batch]

                results[str(key)] = test_accuracy

                if test_accuracy > best_model_accuracy:
                    best_model_accuracy = test_accuracy
                    optimal_parameters = key

    print("Optimal model parameters:")
    print("     eta: ", optimal_parameters[0])
    print("     lambda_reg: ", optimal_parameters[1])
    print("     n_batch: ", optimal_parameters[2])
    print("  Test accuracy: ", best_model_accuracy)


if __name__ == "__main__":
    main()
