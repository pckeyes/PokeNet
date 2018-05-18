# pkmn_load_data_vec.py
#
# DESCRIPTION:
#
# This function opens and reads previously created (pkmn_create_data.py) .txt files that store the matrices and
# arrays containing the X and Y values associated with Pokemon cards pulled from pkmncards.com. The following
# structures are then created:
#
# X         : (n_x, m) - each column is vectorized card img, there are m cards
# X_small   : (n_x_small, m) - each column is a vectorized, scaled-down card img, there are m cards
# Y_name    : (1, m) - list of card names
# Y_type    : (1, m) - list of card types
# Y_set     : (1, m) - list of card sets
# Y_price_l : (1, m) - array of card prices (low)
# Y_price_m : (1, m) - array of card prices (mid)
# Y_price_h : (1, m) - array of card prices (high)
# Y_HP      : (1, m) - array of card HPs
#
# USAGE:
#
# Run this script to load the above data structures, called before training a model. Make sure that dimensions
# specified for loading match those specified during creation (see pkmn_create_data.py)!

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# FN: Convert the image vector back to an Image object (1D array -> 3D rgb Image) (for testing purposes)
def vec2img(img_vec, new_shape):
    img_mat = img_vec.reshape(new_shape[0], new_shape[1], new_shape[2])
    img_mat = Image.fromarray(img_mat, 'RGB')
    return img_mat


def pkmn_load_data_vec(num_cards):

    # User specifies how many cards there are
    m = num_cards
    print("There are %d cards" % num_cards)

    # Define dimensions of images
    n_w, n_h = (600, 824)
    n_w_small, n_h_small = (60, 82)
    n_x = n_h * n_w * 3
    n_x_small = n_h_small * n_w_small * 3

    # Initialize the X and Y structures
    X = np.zeros((n_x, num_cards), dtype=np.uint8)
    X_small = np.zeros((n_x_small, num_cards), dtype=np.uint8)
    Y_name = list()
    Y_type = list()
    Y_set = list()
    Y_price_l = np.zeros((1, num_cards))
    Y_price_m = np.zeros((1, num_cards))
    Y_price_h = np.zeros((1, num_cards))
    Y_HP = np.zeros((1, num_cards))

    # Open .txt files that contain the data
    #f_X = open("X.txt", "r")
    f_X_small = open("X_small.txt", "r")
    f_Y_name = open("Y_name.txt", "r")
    f_Y_type = open("Y_type.txt", "r")
    f_Y_set = open("Y_set.txt", "r")
    f_Y_price_l = open("Y_price_l.txt", "r")
    f_Y_price_m = open("Y_price_m.txt", "r")
    f_Y_price_h = open("Y_price_h.txt", "r")
    f_Y_HP = open("Y_HP.txt", "r")

    # Get the lines (pertaining to columns) for Xs and Ys
    #X_cols = f_X.read().splitlines()
    X_small_cols = f_X_small.read().splitlines()
    Y_name_cols = f_Y_name.read().splitlines()
    Y_type_cols = f_Y_type.read().splitlines()
    Y_set_cols = f_Y_set.read().splitlines()
    Y_price_l_cols = f_Y_price_l.read().splitlines()
    Y_price_m_cols = f_Y_price_m.read().splitlines()
    Y_price_h_cols = f_Y_price_h.read().splitlines()
    Y_HP_cols = f_Y_HP.read().splitlines()

    # For each "column", load the row into the matrices, along with the associated Y labels
    card_i = 0
    for col in range(num_cards):

        print("Loading card " + str(card_i+1) + " out of " + str(num_cards))

        #X_col_array = np.asarray(X_cols[col].split()).reshape((n_x, 1))
        #X[:, col] = X_col_array[:, 0]

        X_small_col_array = np.asarray(X_small_cols[col].split()).reshape((n_x_small, 1))
        X_small[:, col] = X_small_col_array[:, 0]

        Y_name.append(Y_name_cols[col])
        Y_type.append(Y_type_cols[col])
        Y_set.append(Y_set_cols[col])
        Y_price_l[0, col] = np.asarray(Y_price_l_cols[col])
        Y_price_m[0, col] = np.asarray(Y_price_m_cols[col])
        Y_price_h[0, col] = np.asarray(Y_price_h_cols[col])
        if Y_HP_cols[col] != 'None':
            Y_HP[0, col] = np.asarray(Y_HP_cols[col])
        else:
            Y_HP[0, col] = None

        card_i += 1

    print("All cards loaded!")

    # FOR TEST PURPOSES:

    # # Choose a card to display
    # test_i = 0
    # print(Y_name[test_i])
    # print(Y_type[test_i])
    # print(Y_set[test_i])
    # print(Y_price_l[0, test_i])
    # print(Y_price_m[0, test_i])
    # print(Y_price_h[0, test_i])
    # print(Y_HP[0, test_i])
    # test_img_vec = X[:, test_i].astype(np.uint8)
    # test_img = vec2img(test_img_vec, (n_h, n_w, 3))
    # test_imgplot = plt.imshow(test_img)
    # plt.show()

    # Return the outputs
    return X, X_small, Y_type, Y_name, Y_type, Y_set, Y_price_l, Y_price_m, Y_price_h, Y_HP

# FOR TESTING PURPOSES
#pkmn_load_data_vec(3)