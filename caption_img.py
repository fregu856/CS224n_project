"""
- Must be called in one of the following ways:
 $ caption_img.py LSTM (for using the best LSTM model)
 $ caption_img.py GRU (for using te best GRU model)

- Assumes that the image one would like to generate a caption for is called
 "img.jpg" and is placed in the directory "test_img".
"""

import cPickle
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt
import sys

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model
from extract_img_features import extract_img_features

model_type = sys.argv[1]
if model_type not in ["LSTM", "GRU"]:
    raise Exception("Must be called in one of the following ways: \n%s\n%s" %\
                ("$ caption_img.py LSTM", "$ caption_img.py GRU"))

# load the vocabulary:
vocabulary = cPickle.load(open("coco/data/vocabulary"))

# get the img's feature vector:
img_id_2_feature_vector = extract_img_features(["test_img/img.jpg"], demo=True)
feature_vector = img_id_2_feature_vector[0]

# initialize the model:
if model_type == "GRU":
    config = GRU_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = GRU_Model(config, dummy_embeddings, mode="demo")
else:
    config = LSTM_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_Model(config, dummy_embeddings, mode="demo")

# create the saver:
saver = tf.train.Saver()

with tf.Session() as sess:
    # restore the best model:
    if model_type == "GRU":
        saver.restore(sess, "models/GRUs/best_model/model")
    else:
        saver.restore(sess, "models/LSTMs/best_model/model")

    # caption the img (using the best model):
    img_caption = model.generate_img_caption(sess, feature_vector, vocabulary)

# display the img and its generated caption:
I = io.imread("test_img/img.jpg")
plt.imshow(I)
plt.axis('off')
plt.title(img_caption)
plt.show()
