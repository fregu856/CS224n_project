"""
- ASSUMES: that the image dataset has been manually split such that all train
  images are stored in "coco/images/train/", all test images are stored in
  "coco/images/test/" and all val images are stored in "coco/images/val". That
  the Inception-V3 model has been downloaded and placed in inception. That the
  dict numpy_params (containing W_img and b_img taken from the img_transform
  step in a well-performing non-attention model) is placed in
  coco/data/img_features_attention/transform_params.

- DOES: extracts a 64x300 feature array (64 300 dimensional feature vectors,
  one each for 8x8 different img regions) for each train/val/test img and saves
  each individual feature array to disk (to coco/data/img_features_attention).
  Is used in the attention models.
"""

import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import cPickle
from utilities import log

from extract_img_features import load_pretrained_CNN

def extract_img_features_attention(img_paths, demo=False):
    """
    - Runs every image in "img_paths" through the pretrained CNN and
    saves their respective feature array (the third-to-last layer
    of the CNN transformed to 64x300) to disk.
    """

    # load the Inception-V3 model:
    load_pretrained_CNN()

    # load the parameters for the feature vector transform:
    transform_params = cPickle.load(open(
                "coco/data/img_features_attention/transform_params/numpy_params"))
    W_img = transform_params["W_img"]
    b_img = transform_params["b_img"]

    with tf.Session() as sess:
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
                log(str(step))

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
                log("JPEG error for:")
                log(img_path)
                log("******************")
            else:
                if not demo:
                    # get the image id:
                    img_name = img_path.split("/")[3]
                    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
                    img_id = int(img_id)
                else: # (if demo:)
                    # we're only extracting features for one img, (arbitrarily)
                    # set the img id to -:
                    img_id = -1

                # save the img features to disk:
                cPickle.dump(img_features,
                        open("coco/data/img_features_attention/%d" % img_id, "wb"))

def main():
    # define where all val imgs are located:
    val_img_dir = "coco/images/val/"
    # create a list of the paths to all val imgs:
    val_img_paths = [val_img_dir + file_name for file_name in\
                     os.listdir(val_img_dir) if ".jpg" in file_name]

    # define where all test imgs are located:
    test_img_dir = "coco/images/test/"
    # create a list of the paths to all test imgs:
    test_img_paths = [test_img_dir + file_name for file_name in\
                      os.listdir(test_img_dir) if ".jpg" in file_name]

    # define where all train imgs are located:
    train_img_dir = "coco/images/train/"
    # create a list of the paths to all train imgs:
    train_img_paths = [train_img_dir + file_name for file_name in\
                       os.listdir(train_img_dir) if ".jpg" in file_name]

    # create a list of the paths to all imgs:
    img_paths = val_img_paths + test_img_paths + train_img_paths

    # extract all features:
    extract_img_features_attention(img_paths)

if __name__ == '__main__':
    main()
