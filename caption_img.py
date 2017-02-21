"""
- Assumes that the image one would like to generate a caption for is called
 "img.jpg" and is placed in the directory "test_img".
"""

import cPickle
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt

from model import Config, Model
from extract_img_features import extract_img_features

# load the vocabulary:
vocabulary = cPickle.load(open("coco/data/vocabulary"))

# get the img's feature vector:
img_id_2_feature_vector = extract_img_features(["test_img/img.jpg"], demo=True)
feature_vector = img_id_2_feature_vector[0]

# initialize the model:
config = Config()
dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
            dtype=np.float32)
model = Model(config, dummy_embeddings, mode="demo")

# create the saver:
saver = tf.train.Saver()

with tf.Session() as sess:
    # restore the best model:
    saver.restore(sess, "models/LSTMs/best_model/model")

    # caption the img (using the best model):
    img_caption = model.generate_img_caption(sess, feature_vector, vocabulary)

# display the img and its generated caption:
I = io.imread("test_img/img.jpg")
plt.imshow(I)
plt.axis('off')
plt.title(img_caption)
plt.show()
