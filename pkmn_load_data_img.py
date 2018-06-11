# pkmn_load_data_img.py
#
# DESCRIPTION:
#
# This function reads opens and reads previously saved imgs (pkmn_save_cards.py) and created .txt files
# (pkmn_create_data.py) that store the arrays containing the Y values associated with Pokemon cards pulled from
# pkmncards.com. The following structures are then created:
#
# X         : (m, n_H, n_W, 3) - each card (m) is an (n_H x n_W x 3) numpy array
# X_small   : (m, n_H_small, n_W_small, 3) - each card (m) is an (n_H_small x n_W_small x 3) numpy array
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
# Run this function to load the above data structures, called before training a model. Make sure that dimensions
# specified for loading match those specified during creation (see pkmn_create_data.py and pkmn_save_cards.py)!
# Inputs are number of cards (make sure you've created/saved enough data!) and the path to the folder with the imgs

from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import glob


def pkmn_load_data_img(num_cards, card_path):

    # User specifies how many cards there are
    m = num_cards
    print("There are %d cards" % num_cards)

    # Define dimensions of images
    n_w, n_h = (224,224)
    n_w_small, n_h_small = (60, 82)

    # Initialize the X and Y structures
    X = np.zeros((num_cards, n_h, n_w, 3), dtype=np.uint8)
    X_small = np.zeros((num_cards, n_h_small, n_w_small, 3), dtype=np.uint8)
    Y_name = list()
    Y_type = list()
    Y_set = list()
    Y_price_l = np.zeros((1, num_cards))
    Y_price_m = np.zeros((1, num_cards))
    Y_price_h = np.zeros((1, num_cards))
    Y_HP = np.zeros((1, num_cards))

    # Get list of image file names for cards and cards_small
    cards_file_paths = glob.glob(card_path + "*.jpg")
    #cards_small_file_paths = glob.glob(card_path + "/cards_small/*.jpg")

    # Open .txt files that contain the Y data
    f_Y_name = open("labels/Y_name.txt", "r")
    f_Y_type = open("labels/Y_type.txt", "r")
    f_Y_set = open("labels/Y_set.txt", "r")
    f_Y_price_l = open("labels/Y_price_l.txt", "r")
    f_Y_price_m = open("labels/Y_price_m.txt", "r")
    f_Y_price_h = open("labels/Y_price_h.txt", "r")
    f_Y_HP = open("labels/Y_HP.txt", "r")

    # Get the lines (pertaining to columns) for Ys
    Y_name_cols = f_Y_name.read().splitlines()
    Y_type_cols = f_Y_type.read().splitlines()
    Y_set_cols = f_Y_set.read().splitlines()
    Y_price_l_cols = f_Y_price_l.read().splitlines()
    Y_price_m_cols = f_Y_price_m.read().splitlines()
    Y_price_h_cols = f_Y_price_h.read().splitlines()
    Y_HP_cols = f_Y_HP.read().splitlines()

    # For each card, load the row into the matrices, along with the associated Y labels
    for card_i in range(num_cards):

        print("Loading card " + str(card_i+1) + " out of " + str(num_cards))

        X[card_i, :, :, :] = np.array(Image.open(cards_file_paths[card_i]))
        #X_small[card_i, :, :, :] = np.array(Image.open(cards_small_file_paths[card_i]))

        Y_name.append(Y_name_cols[card_i])
        Y_type.append(Y_type_cols[card_i])
        Y_set.append(Y_set_cols[card_i])
        Y_price_l[0, card_i] = np.asarray(Y_price_l_cols[card_i])
        Y_price_m[0, card_i] = np.asarray(Y_price_m_cols[card_i])
        Y_price_h[0, card_i] = np.asarray(Y_price_h_cols[card_i])
        if Y_HP_cols[card_i] != 'nan':
            Y_HP[0, card_i] = np.asarray(Y_HP_cols[card_i])
        else:
            Y_HP[0, card_i] = 0

    print("All cards loaded!")

    # # FOR TEST PURPOSES:
    #
    # # Choose a card to display
    #test_i = 0
    #print(Y_name[test_i])
    #print(Y_type[test_i])
    #print(Y_set[test_i])
    #print(Y_price_l[0, test_i])
    #print(Y_price_m[0, test_i])
    #print(Y_price_h[0, test_i])
    #print(Y_HP[0, test_i])
    #test_imgplot = plt.imshow(X[test_i, :, :, :])
    #plt.show()

    # Return the outputs
    return X, X_small, Y_name, Y_type, Y_set, Y_price_l, Y_price_m, Y_price_h, Y_HP


# FOR TESTING PURPOSES
#pkmn_load_data_img(1, "/home/ubuntu/PokeNet/cards/")
