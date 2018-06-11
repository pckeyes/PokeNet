

# # Pokemon Convnet Classification

import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
import scipy.misc
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import tensorflow as tf
import pkmn_load_data_img as pkmn_data
from sklearn.utils import shuffle
from keras import applications
from keras import optimizers
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)


# Load data
data_path = "/Users/shatzlab/PycharmProjects/Pokemon_Deep_Learning/"
X_loaded, _, _, _, _, Y_loaded, _, _, _ = pkmn_data.pkmn_load_data_img(150, data_path)
Y_loaded = Y_loaded.T

# Dimensions of input data and number of classes
num_cards, n_h, n_w, n_c = X_loaded.shape

# Shuffle X and Y matrices
X_shuffled = X_loaded
Y_shuffled = Y_loaded
shuffled_i = shuffle(range(num_cards))
for i in range(num_cards):
    X_shuffled[i, :, :, :] = X_loaded[shuffled_i[i], :, :, :]
    Y_shuffled[i, 0] = Y_loaded[shuffled_i[i], 0]

# Divide X and Y into train and dev groups
train_end_index = int(0.8 * num_cards) # use 80% of data for train
dev_end_index = int(0.9 * num_cards) # 10% dev (other 10% test)
X_train = X_shuffled[:train_end_index, :, :, :] / 255
X_dev = X_shuffled[train_end_index:dev_end_index, :, :, :] / 255
X_test = X_shuffled[dev_end_index:, :, :, :] / 255
Y_train = Y_shuffled[:train_end_index, :]
Y_dev = Y_shuffled[train_end_index:dev_end_index, :]
Y_test = Y_shuffled[dev_end_index:, :]

m = X_train.shape[0]

print("number of training examples = " + str(X_train.shape[0]))
print("number of dev examples = " + str(X_dev.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_dev shape: " + str(X_dev.shape))
print("Y_dev shape: " + str(Y_dev.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

# Load DenseNet model
model = applications.densenet.DenseNet121(include_top=False, weights = "imagenet", input_shape = (n_w, n_h, 3))
n_layers = len(model.layers)
print('Loaded DenseNet model (ImageNet): ' + str(n_layers) + ' layers')

# Freeze first K layers of model
K = 420
print('Freezing first ' + str(K) + ' layers')
for layer in model.layers[:K]:
    layer.trainable = False

# Add custom layers
x = model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='linear')(x)

# Create final model
model_final = Model(input=model.input, output=predictions)

# Define model params
LR = 0.001
momentum = 0.9
num_epochs = 1
batch_size = 128

# Compile the model
model_final.compile(loss='mean_squared_error', optimizer = optimizers.SGD(lr=LR, momentum=momentum), metrics=['mse'])

# Run the model
history = model_final.fit(X_train, Y_train, epochs = num_epochs, batch_size = batch_size)

# Predict on dev set
preds_train = model_final.evaluate(X_train, Y_train)
preds_dev = model_final.evaluate(X_dev, Y_dev)
preds_test = model_final.evaluate(X_test, Y_test)
print("Train Loss = " + str(preds_train[0]))
print("Train MSE = " + str(preds_train[1]))
print("Dev Loss = " + str(preds_dev[0]))
print("Dev MSE = " + str(preds_dev[1]))
print("Test Loss = " + str(preds_test[0]))
print("Test MSE = " + str(preds_test[1]))
#print(model_final.summary())

