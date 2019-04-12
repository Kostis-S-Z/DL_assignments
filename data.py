import numpy as np
import pickle
from pathlib import Path


parent_dir = str(Path.cwd().parent)  # Get the parent directory of the current working directory
directory = parent_dir + "/cifar-10-batches-py"  # The dataset should be in the parent directory


def load_data(use_all=False, val_size=5000):
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

    if use_all:
        train_data, val_data = load_and_merge(val_size)
    else:
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


def load_and_merge(val_size):
    """
    Load all batches of data. Set aside 5000 samples for validation
    """
    main_file = directory + "/data_batch_"

    file_1 = main_file + "1"
    with open(file_1, 'rb') as fo:
        batch_1 = pickle.load(fo, encoding='bytes')

    data = batch_1[b"data"]
    labels = batch_1[b"labels"]

    # Load all 5 batches
    for i in range(2, 6):
        file = main_file + str(i)

        with open(file, 'rb') as fo:
            batch_i = pickle.load(fo, encoding='bytes')

        data = np.vstack((data, batch_i[b"data"]))
        labels = labels + batch_i[b"labels"]

    # Use the same format of dictionary
    train_data = dict()
    val_data = dict()
    # Use the first X-validation_size data for training and the rest X-validation_size for validation
    data_index = len(data) - val_size
    train_data[b"data"] = data[:data_index]
    train_data[b"labels"] = labels[:data_index]
    val_data[b"data"] = data[data_index:]
    val_data[b"labels"] = labels[data_index:]

    return train_data, val_data


def preprocess_data(data, labels):
    """
    Preprocess data by normalizing between [0,1] and convert labels to one hot
    :return: the preprocessed data
    """
    data = data / 255
    labels = np.eye(10)[labels]

    return data, labels


def process_zero_mean(train, val, test):
    """
    Preprocess data to have a zero mean based on the mean of the training data
    :return: the processed data
    """
    mean = np.mean(train)
    std = np.std(train)
    train_t = (train - mean) / std
    val_t = (val - mean) / std
    test_t = (test - mean) / std
    return train_t, val_t, test_t
