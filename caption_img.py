"""
- Must be called in one of the following ways:
 $ caption_img.py LSTM (for using the best LSTM model)
 $ caption_img.py LSTM_attention (for using the best LSTM_attention model)
 $ caption_img.py GRU (for using te best GRU model)

- Assumes that the image one would like to generate a caption for is called
 "img.jpg" and is placed in the directory "test_img".
"""

import cPickle
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage
import matplotlib.pyplot as plt
import sys

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model
from LSTM_attention_model import LSTM_attention_Config, LSTM_attention_Model
from extract_img_features import extract_img_features
from extract_img_features_attention import extract_img_features_attention

if len(sys.argv) < 2:
    raise Exception("Must be called in one of the following ways: \n%s\n%s" %\
                ("$ caption_random_test_img.py LSTM",
                "$ caption_random_test_img.py LSTM_attention",
                "$ caption_random_test_img.py GRU"))

model_type = sys.argv[1]
if model_type not in ["LSTM", "GRU", "LSTM_attention"]:
    raise Exception("Must be called in one of the following ways: \n%s\n%s" %\
                ("$ caption_random_test_img.py LSTM",
                "$ caption_random_test_img.py LSTM_attention",
                "$ caption_random_test_img.py GRU"))

# load the vocabulary:
vocabulary = cPickle.load(open("coco/data/vocabulary"))

# get the img's features:
if model_type in ["LSTM", "GRU"]:
    img_id_2_feature_vector = extract_img_features(["test_img/img.jpg"], demo=True)
    img_features = img_id_2_feature_vector[0]
elif model_type in ["LSTM_attention"]:
    extract_img_features_attention(["test_img/img.jpg"], demo=True)
    img_features = cPickle.load(
                open("coco/data/img_features_attention/%d" % -1))

# initialize the model:
if model_type == "GRU":
    config = GRU_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = GRU_Model(config, dummy_embeddings, mode="demo")
elif model_type == "LSTM":
    config = LSTM_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_Model(config, dummy_embeddings, mode="demo")
elif model_type == "LSTM_attention":
    config = LSTM_attention_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_attention_Model(config, dummy_embeddings, mode="demo")

# create the saver:
saver = tf.train.Saver()

with tf.Session() as sess:
    # restore the best model:
    if model_type == "GRU":
        saver.restore(sess, "models/GRUs/best_model/model")
    elif model_type == "LSTM":
        saver.restore(sess, "models/LSTMs/best_model/model")
    elif model_type == "LSTM_attention":
        saver.restore(sess, "models/LSTMs_attention/best_model/model")

    # caption the img (using the best model):
    if model_type in ["LSTM", "GRU"]:
        img_caption = model.generate_img_caption(sess, img_features, vocabulary)
    elif model_type in ["LSTM_attention"]:
        img_caption, attention_maps = model.generate_img_caption(sess,
                    img_features, vocabulary)
        # (attention_maps is a list containing caption_length elements, each
        # of which has shape [1, 64, 1])

# display the img and its generated caption:
I = io.imread("test_img/img.jpg")
plt.imshow(I)
plt.axis('off')
plt.title(img_caption)
plt.show()

I_gray = skimage.color.rgb2gray(I)
height, width = I_gray.shape
height_block = int(height/8.)
width_block = int(width/8.)

if model_type in ["LSTM_attention"]:
    for attention_probs, word in zip(attention_maps, img_caption.split(" ")):
        # (attention_probs has shape [1, 64, 1])
        attention_probs = attention_probs.flatten()
        # (attention_probs has shape [64, ])
        attention_probs = np.reshape(attention_probs, (8,8))
        # (attention_probs has shape [8, 8])

        I_test = np.zeros((height, width))
        for i in range(8):
            for j in range(8):
                I_test[i*height_block:(i+1)*height_block, j*width_block:(j+1)*width_block] = np.ones((height_block, width_block))*attention_probs[i,j]
        print I_test
        plt.imshow(0.995*I_test+(1-0.995)*I_gray, cmap="gray")
        plt.axis('off')
        plt.title(img_caption)
        plt.show()
