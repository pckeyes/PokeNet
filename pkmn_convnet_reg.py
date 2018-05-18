# pkmn_convnet_reg.py
#
# DESCRIPTION:
#
# This function trains a CNN to predict scalar values using Pokemon card images as input (such as the card's HP, price).
# The input X is a 4D array containing all of the card images. A linear output / MSE loss are used to train the model.
# Code has been adapted from CS230 assignments. Performance results are saved to a .txt file. The outputs are:
#
# train_RMSE : average error across all predictions in the training set
# dev_RMSE   : average error across all predictions in the dev set
# params     : trained parameters
# costs      : list of costs for each iteration
#
# USAGE:
#
# Run this function with the desired learning_rate, epoch, batch-size (defined at the bottom).
# Make sure to specify how many cards you want trained (load section). Additionally, make sure to define which Y_label
# you want to use (e.g. 'Y_HP', 'Y_price_l') in the load function call (pkmn_load_data_img.py).

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
import pkmn_load_data_img as pkmn_data
from sklearn.utils import shuffle
import random_mini_batches as rmb

# Import data
data_path = "/Users/shatzlab/PycharmProjects/Pokemon_Deep_Learning/"
_, X_loaded, _, _, _, _, _, _, _, Y_loaded = pkmn_data.pkmn_load_data_img(150, data_path)
Y_loaded = Y_loaded.T

# Dimensions of input
num_cards, n_h, n_w, n_c = X_loaded.shape

# Shuffle X and Y matrices
X_shuffled = X_loaded
Y_shuffled = Y_loaded
shuffled_i = shuffle(range(num_cards))
for i in range(num_cards):
    X_shuffled[i, :, :, :] = X_loaded[shuffled_i[i], :, :, :]
    Y_shuffled[i, 0] = Y_loaded[shuffled_i[i], 0]

# Divide X and Y into train and dev groups
train_end_index = int(0.8 * num_cards) #use 80% of data for train
X_train = X_shuffled[:train_end_index, :, :, :] / 255
X_dev = X_shuffled[train_end_index:, :, :, :] / 255
Y_train = Y_shuffled[:train_end_index, :]
Y_dev = Y_shuffled[train_end_index:, :]

m = X_train.shape[0]

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_dev.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_dev.shape))
print("Y_test shape: " + str(Y_dev.shape))


# Create placeholders

def create_placeholders(n_H0, n_W0, n_C0):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, 1] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, 1))

    return X, Y


X, Y = create_placeholders(n_h, n_w, n_c)
print("X = " + str(X))
print("Y = " + str(Y))


# Initialize parameters

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """

    tf.set_random_seed(1)  # so that your "random" numbers match ours

    W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))

    parameters = {"W1": W1,
                  "W2": W2}

    return parameters


# Forward propagation
#
# In detail, we will use the following parameters for all the steps:
#      - Conv2D: stride 1, padding is "SAME"
#      - ReLU
#      - Max pool: Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
#      - Conv2D: stride 1, padding is "SAME"
#      - ReLU
#      - Max pool: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
#      - Flatten the previous output.
#      - FULLYCONNECTED (FC) layer: Apply a fully connected layer (linear activation will included in the cost)

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']

    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')

    # RELU
    A1 = tf.nn.relu(Z1)

    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    P1 = tf.nn.max_pool(A1, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME')

    # CONV2D: filters W2, stride 1, padding 'SAME'
    Z2 = tf.nn.conv2d(P1, W2, strides=[1, 1, 1, 1], padding='SAME')

    # RELU
    A2 = tf.nn.relu(Z2)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tf.nn.max_pool(A2, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    # FLATTEN
    P2 = tf.contrib.layers.flatten(P2)

    # FULLY-CONNECTED without nonlinear activation
    Z3 = tf.contrib.layers.fully_connected(P2, num_outputs=1, activation_fn=None)

    return Z3


# Compute cost

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (1, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """

    cost = tf.reduce_mean(tf.square(Z3 - Y))

    return cost


# Model
#
# The model:
#
# - create placeholders
# - initialize parameters
# - forward propagate
# - compute the cost
# - create an optimizer

def model(X_train, Y_train, X_dev, Y_dev, learning_rate=0.009,
          num_epochs=100, minibatch_size=m, print_cost=True):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep results consistent (tensorflow seed)
    seed = 3  # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape
    costs = []  # To keep track of the cost

    # Create Placeholders of the correct shape
    X, Y = create_placeholders(n_H0, n_W0, n_C0)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = rmb.random_mini_batches(X_train, Y_train, minibatch_size, seed)

            # Peform mini-batch gradient descent
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _, temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # # Plot the cost
        # plt.plot(np.squeeze(costs))
        # plt.ylabel('cost')
        # plt.xlabel('iterations (per tens)')
        # plt.title("Learning rate =" + str(learning_rate))
        # plt.show()

        # Calculate RMSE on the dev set

        # Maybe ROUND to nearest 10 as final output
        rmse = tf.sqrt(tf.reduce_mean(tf.square(Z3 - Y)))
        train_rmse = rmse.eval({X: X_train, Y: Y_train})
        dev_rmse = rmse.eval({X: X_dev, Y: Y_dev})
        print("Train RMSE:", train_rmse)
        print("Dev RMSE:", dev_rmse)

        return train_rmse, dev_rmse, parameters, costs


# Define model hyperparameters
lr = 0.009  # learning rate
num_epochs = 200
mb_size = m  # mini-batch size

# Run it
train_rmse, dev_rmse, parameters, costs = model(X_train, Y_train, X_dev, Y_dev, lr, num_epochs, mb_size)

# Save the performance specs as a .txt file
save_file = open(data_path + "/outputs/convnet_reg_HP_" + str(m) + "_" + str(lr) + "_" + str(num_epochs) + "_" +
                 str(mb_size) + ".txt", "w+")
save_file.write(str(train_rmse) + "\n")
save_file.write(str(dev_rmse) + "\n")
save_file.write(" ".join(map(str, costs)))
save_file.close()