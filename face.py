from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import dlib
import openface
from fr_utils import *
from inception_blocks_v2 import *

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

def triplet_loss(y_true, y_pred, alpha = 0.2):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0],y_pred[1])), axis = None)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(y_pred[0],y_pred[2])), axis = None)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist,neg_dist),alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss,0.0))
    
    return loss
import numpy as np
# FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy'])
load_weights_from_FaceNet(FRmodel)
# database = {}
# database["Tu Vo Van"] = img_to_encoding_source("tu_0.jpg", FRmodel)
# database["An"] = img_to_encoding_source("an_0.jpg", FRmodel)
# database["Hoang"] = img_to_encoding_source("hoang_0.jpg", FRmodel)
# database["Mother Fucker"] = img_to_encoding_source("uy_0.jpg", FRmodel)
# np.save('database.npy', database)

database = np.load('database.npy').item()

def verify(image_path, identity, database, model):
    # Step 1: Compute the encoding for the image. Use img_to_encoding() see example above. (≈ 1 line)
    encoding = img_to_encoding(image_path,model)

    # Step 2: Compute distance with identity's image (≈ 1 line)
    dist = np.linalg.norm(encoding-database[identity])

    # Step 3: Open the door if dist < 0.7, else don't open (≈ 3 lines)
    if dist<0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

# verify("tu2_0.jpg", "Tu Vo Van", database, FRmodel)

def who_is_it(save_path, database, model):
    ## Step 1: Compute the target "encoding" for the image. Use img_to_encoding() see example above. ## (≈ 1 line)
    encoding, image, x, y = img_to_encoding(save_path,model)

    ## Step 2: Find the closest encoding ##
    min_dist = 100

    # Loop over the database dictionary's names and encodings.
    for (name, db_enc) in database.items():

        # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
        dist = np.linalg.norm(encoding-db_enc)

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
        if dist<min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
        cv2.putText(image, "Not Recognized!!", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, [0, 255, 0], 3)
        cv2.imshow('who??', image)
        cv2.waitKey(0)
    else:
        print ("it's " + str(identity) + ", the distance is " + str(min_dist))
        cv2.putText(image, str(identity), (y, x), cv2.FONT_HERSHEY_SIMPLEX, 1, [0,255,0], 3)
        cv2.imshow('who??', image)
        cv2.waitKey(0)

    return min_dist, identity

who_is_it("uy2.jpg", database, FRmodel)
# # cv2.imshow('who??', cv2.imread("tu.jpg"))
# cv2.waitKey(0)
