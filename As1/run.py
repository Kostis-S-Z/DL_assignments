"""
Created by Kostis S-Z @ 2019-03-27
"""

import numpy as np
import matplotlib as plt
from network import OneLayerNetwork


directory = "cifar-10-batches-py"

model_parameters = {
    "eta": 0.1,
    "n_batch": 20,
    "min_delta": 0.01,
    "patience": 10
}


def main():
    train_x, train_y, val_x, val_y, test_x, test_y = load_data()

    train_x, train_y = preprocess_data(train_x, train_y)
    val_x, val_y = preprocess_data(val_x, val_y)
    test_x, test_y = preprocess_data(test_x, test_y)

    net = OneLayerNetwork(**model_parameters)

    net.train(train_x, train_y, val_x, val_y, n_epochs=2, early_stop=True, verbose=True)

    net.test(test_x, test_y)


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
    # labels = np.eye(10)[labels]

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
