# DL_assignments
Deep Learning DD2424 @ KTH 2019 Assignments

### Assignment 1: Building a Neural Network from scratch

The goal of this assignment was to build a Single Layer Neural Network from scratch and train it on the popular image dataset CIFAR-10. The learning algorithm used was mini-batch gradient descent.

##### Features:

- Choice between Cross-entropy and SVM Multi-class loss function
- L2 Regularisation
- Early stopping
- Shuffling data samples every epoch
- Grid Search for optimal hyper parameter tuning
- Plot loss progress & learnt class representations (weight matrices)


### Assignment 2: Expanding to a Two-Layer Neural Network

Next, we expand the network to use two layers and cyclical learning rates.

##### Features:

- Pre-process data to a zero mean
- Cyclical Learning rates
- Extensive search for optimal value of regularisation term


### Assignment 3: Expanding (even more!) to a Multi-Layer Neural Network + BatchNormalisation

##### Features:

- Easy way to setup whole structure of network
- Batch Normalization: 
Method to reduce the dependency of the network from the initialization values and be able to use a higher learning rate.
It normalizes the activations in a standard normal distribution (with zero mean and 1 std) and using learnable parameters (gamma and beta) and a linear transformation, it enables us to learn what distribution fits most for the activations of the layers.
This will stop the problem of vanishing / exploding gradients as the outputs will no longer be saturated. 

