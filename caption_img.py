"""
- Must be called in one of the following ways:
 $ caption_img.py LSTM (for using the best LSTM model)
 $ caption_img.py LSTM_attention (for using the best LSTM_attention model)
 $ caption_img.py GRU (for using the best GRU model)
 $ caption_img.py GRU_attention (for using the best GRU_attention model)

- ASSUMES: that preprocess_captions.py has already been run. That the image one
  would like to generate a caption for is called "img.jpg" and is placed in the
  directory "img_to_caption". That the weights for the best
  LSTM/GRU/LSTM_attention/GRU_attention model has been placed in
  models/**model_type**/best_model with names model.filetype.

- DOES: generates a caption for "img.jpg" using the best model of the specified
  model type and displays the img and its caption. For attention models, it also
  displays a figure visualizing the img attention at the time of prediciton for
  each word in the caption.
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
from GRU_attention_model import GRU_attention_Config, GRU_attention_Model
from extract_img_features import extract_img_features
from extract_img_features_attention import extract_img_features_attention

# check that the script was called in a valid way:
if len(sys.argv) < 2:
    raise Exception("Must be called in one of the following ways: \n%s\n%s\n%s\n%s" %\
                ("$ caption_img.py LSTM",
                "$ caption_img.py LSTM_attention",
                "$ caption_img.py GRU",
                "$ caption_img.py GRU_attention"))

model_type = sys.argv[1]
if model_type not in ["LSTM", "GRU", "LSTM_attention", "GRU_attention"]:
    raise Exception("Must be called in one of the following ways: \n%s\n%s\n%s\n%s" %\
                ("$ caption_img.py LSTM",
                "$ caption_img.py LSTM_attention",
                "$ caption_img.py GRU",
                "$ caption_img.py GRU_attention"))

# load the vocabulary:
vocabulary = cPickle.load(open("coco/data/vocabulary"))

# get the img's features:
if model_type in ["LSTM", "GRU"]:
    img_id_2_feature_vector = extract_img_features(["img_to_caption/img.jpg"], demo=True)
    img_features = img_id_2_feature_vector[0]
elif model_type in ["LSTM_attention", "GRU_attention"]:
    extract_img_features_attention(["img_to_caption/img.jpg"], demo=True)
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
elif model_type == "GRU_attention":
    config = GRU_attention_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = GRU_attention_Model(config, dummy_embeddings, mode="demo")

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
    elif model_type == "GRU_attention":
        saver.restore(sess, "models/GRUs_attention/best_model/model")

    # caption the img (using the best model):
    if model_type in ["LSTM", "GRU"]:
        img_caption = model.generate_img_caption(sess, img_features, vocabulary)
    elif model_type in ["LSTM_attention", "GRU_attention"]:
        img_caption, attention_maps = model.generate_img_caption(sess,
                    img_features, vocabulary)
        # (attention_maps is a list containing caption_length elements, each
        # of which has shape [1, 64, 1])

# display the img and its generated caption:
I = io.imread("img_to_caption/img.jpg")
plt.figure(1)
plt.imshow(I)
plt.axis('off')
plt.title(img_caption)

# for attention models, also display a figure visualizing the img attention for
# each word in the caption:
if model_type in ["LSTM_attention", "GRU_attention"]:
    # get a gray scale version of the img:
    I_gray = skimage.color.rgb2gray(I)
    # get some img paramaters:
    height, width = I_gray.shape
    height_block = int(height/8.)
    width_block = int(width/8.)
    # turn the caption into a vector of the words:
    img_caption_vector = img_caption.split(" ")
    caption_length = len(img_caption_vector)

    plt.figure(2)

    # create a plot with an img for each word in the generated caption,
    # visualizing the img attention when the word was generated:
    if int(caption_length/3.) == caption_length/3.:
        no_of_rows = int(caption_length/3.)
    else:
        no_of_rows = int(caption_length/3.) + 1

    for step, (attention_probs, word) in\
                enumerate(zip(attention_maps, img_caption_vector)):
        plt.subplot(no_of_rows, 3, step+1)
        # flatten the attention_probs from shape [1, 64, 1] to [64, ]:
        attention_probs = attention_probs.flatten()
        # reshape the attention_probs to shape [8,8]:
        attention_probs = np.reshape(attention_probs, (8,8))

        # convert the 8x8 attention probs map to an img of the same size as the img:
        I_att = np.zeros((height, width))
        for i in range(8):
            for j in range(8):
                I_att[i*height_block:(i+1)*height_block, j*width_block:(j+1)*width_block] =\
                            np.ones((height_block, width_block))*attention_probs[i,j]

        # blend the grayscale img and the attention img:
        alpha = 0.97
        I_blend = alpha*I_att+(1-alpha)*I_gray
        # display the blended img:
        plt.imshow(I_blend, cmap="gray")
        plt.axis('off')
        plt.title(word)

plt.show()
