"""
- Assumes that the image dataset has been manually split such that all train
  images are stored in "coco/images/train/", all test images are stored in
  "coco/images/test/" and all val images are stored in "coco/images/val".
"""

import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle

from LSTM_model import LSTM_Config, LSTM_Model
from extract_img_features import load_pretrained_CNN

def extract_img_features_attention(img_paths, demo=False):
    """
    -
    """

    # load the Inception-V3 model:
    load_pretrained_CNN()

    # initialize the model containing W_img and b_img:
    config = LSTM_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_Model(config, dummy_embeddings, mode="demo")

    # create the saver:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore all model variables:
        params_dir = "coco/data/img_features_attention/transform_params"
        saver.restore(sess, "%s/model" % params_dir)

        # get the restored W_img and b_img:
        with tf.variable_scope("img_transform", reuse=True):
            W_img = tf.get_variable("W_img")
            b_img = tf.get_variable("b_img")

        # get the third-to-last layer in the Inception-V3 model (a tensor
        # of shape (1, 8, 8, 2048)):
        img_features_tensor = sess.graph.get_tensor_by_name("mixed_10/join:0")
        # reshape the tensor to shape (64, 2048):
        img_features_tensor = tf.reshape(img_features_tensor, (64, 2048))

        # apply the img transorm (get a tensor of shape (64, 300)):
        linear_transform = tf.matmul(img_features_tensor, W_img) + b_img
        img_features_tensor = tf.nn.sigmoid(linear_transform)

        for step, img_path in enumerate(img_paths):
            if step % 10 == 0:
                print step

            # read the image:
            img_data = gfile.FastGFile(img_path, "rb").read()
            try:
                # get the img features (np array of shape (64, 300)):
                img_features = sess.run(img_features_tensor,
                        feed_dict={"DecodeJpeg/contents:0": img_data})
            except:
                print "JPEG error for:"
                print img_path
                print "******************"
            else:
                if not demo:
                    # get the image id:
                    img_name = img_path.split("/")[3]
                    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
                    img_id = int(img_id)
                else:
                    img_id = 0

                # save the img features to disk:
                cPickle.dump(img_features,
                        open("coco/data/img_features_attention/%d" % img_id, "wb"))

def main():
    # define where all val imgs are located:
    val_img_dir = "coco/images/val/"
    # create a list of the paths to all val imgs:
    val_img_paths = [val_img_dir + file_name for file_name in\
                     os.listdir(val_img_dir) if ".jpg" in file_name]
    # get the features for all val imgs:
    extract_img_features_attention(val_img_paths)
    # save on disk:
    print "val done!"

    # define where all test imgs are located:
    test_img_dir = "coco/images/test/"
    # create a list of the paths to all test imgs:
    test_img_paths = [test_img_dir + file_name for file_name in\
                      os.listdir(test_img_dir) if ".jpg" in file_name]
    # get the features for all test imgs:
    extract_img_features_attention(test_img_paths)
    print "test done!"

    # define where all train imgs are located:
    train_img_dir = "coco/images/train/"
    # create a list of the paths to all train imgs:
    train_img_paths = [train_img_dir + file_name for file_name in\
                       os.listdir(train_img_dir) if ".jpg" in file_name]
    # get the features for all train imgs:
    extract_img_features_attention(train_img_paths)

if __name__ == '__main__':
    main()
