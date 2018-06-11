# pkmn_create_data.py
#
# DESCRIPTION:
#
# This script reads in cards.json, which stores key:value pairs scraped from Pokemon cards (pkmncards.com),
# and writes the following .txt files:
#
# X.txt         : each row/line is a vectorized representation of a pkmn card (white-spaced separated values)
#                 m cards (lines), each with n_x = n_h x n_w x 3 (rgb) values
# X_small.txt   : same as X.txt, but cards are reduced in size, therefore vector lengths are shorter
#                 m cards (lines), each with n_x_small values
# Y_name.txt    : each row/line is the string value of the 'name' of the card
#                 m cards (lines)
# Y_type.txt    : each row/line is the string value of the 'type' of the card
#                 m cards (lines)
# Y_set.txt     : each row/line is the string value of the 'set' the card belongs to
#                 m cards (lines)
# Y_price_l.txt : each row/line is the value of the 'low price' for the card
#                 m cards (lines)
# Y_price_m.txt : each row/line is the value of the 'mid price' for the card
#                 m cards (lines)
# Y_price_h.txt : each row/line is the value of the 'high price' for the card
#                 m cards (lines)
# Y_HP.txt      : each row/line is the value of the 'HP' of the card
#                 m cards (lines)
# USAGE:
#
# Run this script once to generate all of the .txt files (later use pkmn_load_data.py to load the data
# before each use). User defines the reshaped dimensions of cards (n_h, n_w), and for the scaled down cards
# (n_h_small, n_w_small) (note: not all images from website are same size, so all will be reshaped).

import json
import urllib.request
import io
from PIL import Image
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


# FN: Convert the Image object into a vector (3D rgb Image -> 1D array)
def img2vec(img):
    img_arr = np.array(img)
    img_vec = img_arr.reshape(img_arr.shape[0] * img_arr.shape[1] * 3, 1)
    return img_vec


# FN: Convert the image vector back to an Image object (1D array -> 3D rgb Image)
def vec2img(img_vec, new_shape):
    img_mat = img_vec.reshape(new_shape[0], new_shape[1], new_shape[2])
    img_mat = Image.fromarray(img_mat, 'RGB')
    return img_mat


# SCRIPT START

# Load the .json file as a dict
with open('/home/ubuntu/PokeNet/PkmnCardScraper/cards.json') as data_file:
    data = json.load(data_file)
# How many cards are there?
num_cards = len(data)
m = num_cards
print("There are %d cards" % num_cards)

# Define the dimensions of the cards and small-cards
n_w, n_h = (224, 224)
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

# Iterate through all cards, copying values from dict to structures
#data = data[7700:7720]
xi = 0
for card in data:

    print("Vectorizing card " + str(xi+1) + " out of " + str(num_cards))

    # Get url for current card, open img file, reshape it, and vectorize
    cur_url = card['img']
    with urllib.request.urlopen(cur_url) as url:
        cur_f = io.BytesIO(url.read())
    cur_img_og = Image.open(cur_f).convert('RGB')
    cur_img = cur_img_og.resize((n_w, n_h))
    #cur_img_small = cur_img_og.resize((n_w_small, n_h_small))
    cur_img_vec = img2vec(cur_img)
    #cur_img_vec_small = img2vec(cur_img_small)

    # Assign vectors and other values to superstructures
    X[:, xi] = cur_img_vec[:, 0]
    #X_small[:, xi] = cur_img_vec_small[:, 0]
    Y_name.append(card['name'])
    Y_type.append(card['type'])
    Y_set.append(card['set'])
    #remove commas from prices
    card['low price'] = card['low price'].replace(',','')
    card['mid price'] = card['mid price'].replace(',','')
    card['high price'] = card['high price'].replace(',','')
    Y_price_l[0, xi] = card['low price']
    Y_price_m[0, xi] = card['mid price']
    Y_price_h[0, xi] = card['high price']
    if card['HP'].isnumeric():
        Y_HP[0, xi] = card['HP']
    else:
        Y_HP[0, xi] = None

    xi += 1

print("All cards vectorized!")

# Write the data structures

# Write X
f_X = open("X.txt", "w+")
for col in range(num_cards):
    cur_col = X[:, col].astype(np.uint8)
    cur_col_list = cur_col.tolist()
    towrite = " ".join(map(str, cur_col_list))
    f_X.write(towrite + "\n")
f_X.close()

# Write X_small
#f_X_small = open("X_small.txt", "w+")
#for col in range(num_cards):
#    cur_col = X_small[:, col].astype(np.uint8)
#    cur_col_list = cur_col.tolist()
#    towrite = " ".join(map(str, cur_col_list))
#    f_X_small.write(towrite + "\n")
#f_X_small.close()

# Write Y_name
f_Y_name = open("Y_name.txt", "w+")
for name in Y_name:
    f_Y_name.write(name + "\n")
f_Y_name.close()

# Write Y_type
f_Y_type = open("Y_type.txt", "w+")
for type in Y_type:
    f_Y_type.write(type + "\n")
f_Y_type.close()

# Write Y_set
f_Y_set = open("Y_set.txt", "w+")
for set in Y_set:
    f_Y_set.write(set + "\n")
f_Y_set.close()

# Write Y_price_l
f_Y_price_l = open("Y_price_l.txt", "w+")
for col in range(num_cards):
    price_l = Y_price_l[0, col]
    f_Y_price_l.write(str(price_l) + "\n")
f_Y_price_l.close()

# Write Y_price_m
f_Y_price_m = open("Y_price_m.txt", "w+")
for col in range(num_cards):
    price_m = Y_price_m[0, col]
    f_Y_price_m.write(str(price_m) + "\n")
f_Y_price_m.close()

# Write Y_price_h
f_Y_price_h = open("Y_price_h.txt", "w+")
for col in range(num_cards):
    price_h = Y_price_h[0, col]
    f_Y_price_h.write(str(price_h) + "\n")
f_Y_price_h.close()

# Write Y_HP
f_Y_HP = open("Y_HP.txt", "w+")
for col in range(num_cards):
    HP = Y_HP[0, col]
    f_Y_HP.write(str(HP) + "\n")
f_Y_HP.close()

# # FOR TESTING PURPOSES:
#
# # Test a specific card (index set by test_i)
# test_i = 0
#
# # Pull out vec from X (column), then convert to image and display
# test_img_vec = X[:, test_i].astype(np.uint8)
# test_img = vec2img(test_img_vec, (n_h, n_w, 3))
# test_imgplot = plt.imshow(test_img)
# plt.show()
#
# # Pull out vec from X_short (column), then convert to image and display
# test_img_small_vec = X_small[:, test_i].astype(np.uint8)
# test_img_small = vec2img(test_img_small_vec, (n_h_small, n_w_small, 3))
# test_imgplot_small = plt.imshow(test_img_small)
# plt.show()
#
# # Pull out labeled Y values for this card
# print(Y_name[test_i])
# print(Y_type[test_i])
# print(Y_set[test_i])
# print(Y_price_l[0, test_i])
# print(Y_price_m[0, test_i])
# print(Y_price_h[0, test_i])
# print(Y_HP[0, test_i])
