# pkmn_save_cards.py
#
# DESCRIPTION:
#
# This script reads a .JSON file (data scraped from pokemoncards.com) to search through pokemon card urls, opens the
# image files, resizes them (large and small), and saves them as .JPEG files.
#

# USAGE:
#
# Run this script once before model training. It will create all the large and small versions of the images. Then use
# the pkmn_load_data_img.py function to read in the img files into memory


import json
import urllib.request
import io
from PIL import Image


# SCRIPT START

# Save path
save_path = "/Users/shatzlab/PycharmProjects/Pokemon_Deep_Learning/"

# Load the .json file as a dict
with open('cards.json') as data_file:
    data = json.load(data_file)

# Only use a subset of the data?
data = data[150:300]

# How many cards are there?
num_cards = len(data)
m = num_cards
print("There are %d cards" % num_cards)

# Define the dimensions of the cards and small-cards
n_w, n_h = (600, 824)
n_w_small, n_h_small = (60, 82)
n_x = n_h * n_w * 3
n_x_small = n_h_small * n_w_small * 3

# Iterate through all cards, opening and saving the img file
xi = 150
for card in data:

    print("Saving card " + str(xi+1) + " out of " + str(num_cards))

    # Get url for current card, open img file, reshape it, and save it
    cur_url = card['img']
    with urllib.request.urlopen(cur_url) as url:
        cur_f = io.BytesIO(url.read())
    cur_img_og = Image.open(cur_f).convert('RGB')
    cur_img = cur_img_og.resize((n_w, n_h))
    cur_img_small = cur_img_og.resize((n_w_small, n_h_small))

    cur_img.save(save_path + "cards/" + str(xi) + ".jpg", 'JPEG')
    cur_img_small.save(save_path + "cards_small/" + str(xi) + "_small.jpg", 'JPEG')
    xi += 1

print("All cards saved!")

