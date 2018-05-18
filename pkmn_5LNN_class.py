#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 10:26:17 2018

@author: piperkeyes
"""
#DESCRIPTION:
#This script trains a 5 layer NN to classify images of Pokemon cards based
#on the type of the card. The input X is a 2D matrix of the shape (nx, m), 
#where nx is the number of pixels in the image and m is the number of images
#loaded. Pass the desired number of images into the pkmn_load_data_vec() method.
#The output layer uses a softmax activation function and the cost
#function uses cross entropy loss. This code was adapted from the CS230
#Tensorflow tutorial assignment.
#
# train_accuracy : percentage of correctly classified cards in the training set
# dev_accuracy   : percentage of correctly classified cards in the dev set
# params         : trained parameters
# costs          : list of costs for each iteration

import tensorflow as tf
#from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
import numpy as np
import pkmn_load_data_vec as pkmn_data
from sklearn.utils import shuffle

#Import and normalize data
#currently set to load type as Y vector
_, X, Y_type, _, _, _, _, _, _, _ = pkmn_data.pkmn_load_data_vec(200)
X = X/255
n_y = 12 #total number of classes

#Convert types from text to class vectors
grass_label = [1,0,0,0,0,0,0,0,0,0,0,0]
fire_label = [0,1,0,0,0,0,0,0,0,0,0,0]
water_label = [0,0,1,0,0,0,0,0,0,0,0,0]
lightning_label = [0,0,0,1,0,0,0,0,0,0,0,0]
psychic_label = [0,0,0,0,1,0,0,0,0,0,0,0]
fighting_label = [0,0,0,0,0,1,0,0,0,0,0,0]
darkness_label = [0,0,0,0,0,0,1,0,0,0,0,0]
metal_label = [0,0,0,0,0,0,0,1,0,0,0,0]
fairy_label = [0,0,0,0,0,0,0,0,1,0,0,0]
dragon_label = [0,0,0,0,0,0,0,0,0,1,0,0]
colorless_label = [0,0,0,0,0,0,0,0,0,0,1,0]
na_label = [0,0,0,0,0,0,0,0,0,0,0,1]
types = ['Grass', 'Fire', 'Water', 'Lightning', 'Psychic', 'Fighting', 'Darkness', 'Metal', 'Fairy', 'Dragon', 'Colorless', 'N/A']
type_labels = [grass_label, fire_label, water_label, lightning_label, psychic_label, fighting_label, darkness_label, metal_label, fairy_label, dragon_label, colorless_label, na_label]

Y_type_vectorized = np.zeros((X.shape[1], n_y))
iter = 0
for pkmn_type in Y_type:
    if pkmn_type == 'Grass': Y_type_vectorized[iter,:] = grass_label
    elif pkmn_type == 'Fire': Y_type_vectorized[iter,:] = fire_label
    elif pkmn_type == 'Water': Y_type_vectorized[iter,:] = water_label
    elif pkmn_type == 'Lightning': Y_type_vectorized[iter,:] = lightning_label
    elif pkmn_type == 'Psychic': Y_type_vectorized[iter,:] = psychic_label
    elif pkmn_type == 'Fighting': Y_type_vectorized[iter,:] = fighting_label
    elif pkmn_type == 'Darkness': Y_type_vectorized[iter,:] = darkness_label
    elif pkmn_type == 'Metal': Y_type_vectorized[iter,:] = metal_label
    elif pkmn_type == 'Fairy': Y_type_vectorized[iter,:] = fairy_label
    elif pkmn_type == 'Dragon': Y_type_vectorized[iter,:] = dragon_label
    elif pkmn_type == 'Colorless': Y_type_vectorized[iter,:] = colorless_label
    elif pkmn_type == 'N/A': Y_type_vectorized[iter,:] = na_label
    #If card has two types, choose primary (i.e. first listed) type
    else:
        first_type, _ = pkmn_type.split(',')
        for j in range(len(types)):
            if first_type == types[j]: Y_type_vectorized[iter,:] = type_labels[j]
    iter += 1
Y_type_vectorized = Y_type_vectorized.T

##Test that all cards have only one label
#count = 0
#for col in range(len(Y_type_vectorized[0])):
#    col_sum = np.sum(Y_type_vectorized[:,col])
#    if col_sum != 1: 
#        count += 1

#Randomize X and Y matrices
X_shuffled, Y_type_vectorized_shuffled = shuffle(X.T, Y_type_vectorized.T)
X_shuffled = X_shuffled.T
Y_type_vectorized_shuffled = Y_type_vectorized_shuffled.T

#Divide X and Y into train and dev groups
train_end_index = int(0.8 * X_shuffled.shape[1]) #use 80% of data for train
X_train = X_shuffled[:,:train_end_index]
X_dev = X_shuffled[:,train_end_index:]
n_x = X_train.shape[0]
Y_train = Y_type_vectorized_shuffled[:,:train_end_index]
Y_dev = Y_type_vectorized_shuffled[:,train_end_index:]

def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, shape = (n_x, None), name = 'X')
    Y = tf.placeholder(tf.float32, shape = (n_y, None), name = 'Y')
    return X, Y

def initialize_parameters(n_x):
    W1 = tf.get_variable('W1', [512, n_x], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [512, 1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable('W2', [512, 512], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', [512, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable('W3', [256, 512], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', [256, 1], initializer = tf.zeros_initializer())
    W4 = tf.get_variable('W4', [128, 256], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable('b4', [128, 1], initializer = tf.zeros_initializer())
    W5 = tf.get_variable('W5', [12, 128], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable('b5', [12, 1], initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}
    
    return parameters

def forward_propogation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    
    #perform linear -> relu until generating Z matrix of output layer
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3
    A3 = tf.nn.relu(Z3)
    Z4 = tf.matmul(W4, A3) + b4
    A4 = tf.nn.relu(Z4)
    Z5 = tf.matmul(W5, A4) + b5
    
    return Z5

def compute_cost(Z5, Y):
    logits = tf.transpose(Z5)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

def model(X_train, Y_train, X_dev, Y_dev, learning_rate = 0.0005, num_epochs = 150,  print_cost = True):
    ops.reset_default_graph()
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []
    
    #Create placeholders
    X, Y = create_placeholders(n_x, n_y)
    
    #Initialize params
    parameters = initialize_parameters(n_x)
    
    #Run forward prop
    Z5 = forward_propogation(X, parameters)
    
    #compute cost
    cost = compute_cost(Z5, Y)
    
    #Run back prop
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    #Initialize all variables
    init = tf.global_variables_initializer()
    
    #Start session
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0
            
            #TODO: implement minibatches
            
            _ , curr_cost = sess.run([optimizer, cost], feed_dict = {X: X_train, Y:Y_train})
            epoch_cost += curr_cost
            
            # Print the cost every epoch
            if print_cost == True and epoch % 10 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        
        #Plot cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        #Save trained parameters
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z5), tf.argmax(Y))

        # Calculate accuracy on the dev set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Dev Accuracy:", accuracy.eval({X: X_dev, Y: Y_dev}))
        
        return parameters
    
#Train the model
params = model(X_train, Y_train, X_dev, Y_dev)