import json
import urllib.request
import io
from PIL import Image
import numpy as np
#import matplotlib.pyplot as plt


def img2vec(img):
    img_arr = np.array(img)
    img_vec = img_arr.reshape(img_arr.shape[0] * img_arr.shape[1] * img_arr.shape[2], 1)
    return img_vec


def vec2img(img_vec, new_shape):
    img_mat = img_vec.reshape(new_shape[0], new_shape[1], new_shape[2])
    img_mat = Image.fromarray(img_mat, 'RGB')
    return img_mat


with open('cards.json') as data_file:
    data = json.load(data_file)

num_cards = len(data)
print("There are %d cards" % num_cards)

n_w, n_h = (3, 4)#(600, 824)
n_w_small, n_h_small = (60, 82)#(150, 206)

n_x = n_h * n_w * 3
n_x_small = n_h_small * n_w_small * 3

X = np.zeros((n_x, num_cards), dtype=np.uint8)
X_small = np.zeros((n_x_small, num_cards), dtype=np.uint8)

Y_name = list()
Y_type = list()
Y_set = list()
Y_price_l = np.zeros((1, num_cards))
Y_price_m = np.zeros((1, num_cards))
Y_price_h = np.zeros((1, num_cards))
Y_HP = np.zeros((1, num_cards))

xi = 0

for card in data:

    print("Vectorizing card " + str(xi+1) + " out of " + str(num_cards))
    cur_url = card['img']
    with urllib.request.urlopen(cur_url) as url:
        cur_f = io.BytesIO(url.read())
    cur_img = Image.open(cur_f)
    cur_img = cur_img.resize((n_w, n_h))
    cur_img_small = cur_img.resize((n_w_small, n_h_small))
    cur_img_vec = img2vec(cur_img)
    cur_img_vec_small = img2vec(cur_img_small)

    X[:, xi] = cur_img_vec[:, 0]
    X_small[:, xi] = cur_img_vec_small[:, 0]

    Y_name.append(card['name'])
    Y_type.append(card['type'])
    Y_set.append(card['set'])
    Y_price_l[0, xi] = card['low price']
    Y_price_m[0, xi] = card['mid price']
    Y_price_h[0, xi] = card['high price']
    if card['HP'].isnumeric():
        Y_HP[0, xi] = card['HP']
    else:
        Y_HP[0, xi] = None

    xi += 1

print("All cards vectorized!")

#print(X)
#
## Write the data structures
#
## Write X
#f_X = open("X.txt", "w+")
#for col in range(num_cards):
#    cur_col = X[:, col].astype(np.uint8)
#    cur_col_list = cur_col.tolist()
#    towrite = " ".join(map(str, cur_col_list))
#    f_X.write(towrite + "\n")
#f_X.close()
#
## Write X_small
#f_X_small = open("X_small.txt", "w+")
#for col in range(num_cards):
#    cur_col = X_small[:, col].astype(np.uint8)
#    cur_col_list = cur_col.tolist()
#    towrite = " ".join(map(str, cur_col_list))
#    f_X_small.write(towrite + "\n")
#f_X_small.close()
#
## Write Y_name
#f_Y_name = open("Y_name.txt", "w+")
#for name in Y_name:
#    f_Y_name.write(name + "\n")
#f_Y_name.close()
#
## Write Y_type
#f_Y_type = open("Y_type.txt", "w+")
#for type in Y_type:
#    f_Y_type.write(type + "\n")
#f_Y_type.close()
#
## Write Y_set
#f_Y_set = open("Y_set.txt", "w+")
#for set in Y_set:
#    f_Y_set.write(set + "\n")
#f_Y_set.close()
#
## Write Y_price_l
#f_Y_price_l = open("Y_price_l.txt", "w+")
#for col in range(num_cards):
#    price_l = Y_price_l[0, col]
#    f_Y_price_l.write(str(price_l) + "\n")
#f_Y_price_l.close()
#
## Write Y_price_m
#f_Y_price_m = open("Y_price_m.txt", "w+")
#for col in range(num_cards):
#    price_m = Y_price_m[0, col]
#    f_Y_price_m.write(str(price_m) + "\n")
#f_Y_price_m.close()
#
## Write Y_price_h
#f_Y_price_h = open("Y_price_h.txt", "w+")
#for col in range(num_cards):
#    price_h = Y_price_h[0, col]
#    f_Y_price_h.write(str(price_h) + "\n")
#f_Y_price_h.close()
#
## Write Y_HP
#f_Y_HP = open("Y_HP.txt", "w+")
#for col in range(num_cards):
#    HP = Y_HP[0, col]
#    f_Y_HP.write(str(HP) + "\n")
#f_Y_HP.close()
#
## For testing purposes
##test_i = 5;
#
## Pull out vec from X (vec), then convert to image and display
##test_img_vec = X[:, test_i].astype(np.uint8)
##test_img = vec2img(test_img_vec, (n_h, n_w, 3))
##test_imgplot = plt.imshow(test_img)
##plt.show()
#
## Pull out vec from X_short (vec), then convert to image and display
##test_img_small_vec = X_small[:, test_i].astype(np.uint8)
##test_img_small = vec2img(test_img_small_vec, (n_h_small, n_w_small, 3))
##test_imgplot_small = plt.imshow(test_img_small)
##plt.show()
#
## Pull out labeled Y values
##print(Y_name[test_i])
##print(Y_type[test_i])
##print(Y_set[test_i])
##print(Y_price_l[0, test_i])
##print(Y_price_m[0, test_i])
##print(Y_price_h[0, test_i])
##print(Y_HP[0, test_i])
