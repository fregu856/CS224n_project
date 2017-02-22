# NOTE! val should be replaced by test once the project is finished!

"""
- Must be called in one of the following ways:
 $ caption_random_test_img.py LSTM (for using the best LSTM model)
 $ caption_random_test_img.py GRU (for using te best GRU model)

- Assumes that the image one would like to generate a caption for is called
 "img.jpg" and is placed in the directory "test_img".
"""

import cPickle
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
import matplotlib.pyplot as plt

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/home/fregu856/CS224n/project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model

model_type = sys.argv[1]
if model_type not in ["LSTM", "GRU"]:
    raise Exception("Must be called in one of the following ways: \n%s\n%s" %\
                ("$ caption_random_test_img.py LSTM",
                "$ caption_random_test_img.py GRU"))

# load all needed data:
val_img_ids = cPickle.load(open("coco/data/val_img_ids"))
val_img_id_2_feature_vector =\
            cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
vocabulary = cPickle.load(open("coco/data/vocabulary"))

# pick a random test img:
random.shuffle(val_img_ids)
img_id = int(val_img_ids[0])

# get the img's feature vector:
feature_vector = val_img_id_2_feature_vector[img_id]

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

# get the img's file name:
true_captions_file = "coco/annotations/captions_val2014.json"
coco = COCO(true_captions_file)
img = coco.loadImgs(img_id)[0]
img_file_name = img["file_name"]

# display the img and its generated caption:
I = io.imread("coco/images/val/%s" % img_file_name)
plt.imshow(I)
plt.axis('off')
plt.title(img_caption)
plt.show()
