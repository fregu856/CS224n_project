"""
- Assumes that the image dataset has been manually split such that all train
  images are stored in "coco/images/train/", all test images are stored in
  "coco/images/test/" and all val images are stored in "coco/images/val".

- Assumes that "split_img_ids.py" already has been run.
"""

import os
import re
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pyplot as plt
import cPickle

# define where the pretrained inception model is located:
model_dir = "inception"

def load_pretrained_CNN():
    """
    - Loads the pretrained Inception-V3 model.
    """

    path_to_saved_model = os.path.join(model_dir,
            "classify_image_graph_def.pb")

    with gfile.FastGFile(path_to_saved_model, "rb") as model_file:
        # create an empty GraphDef object:
        graph_def = tf.GraphDef()

        # import the model definitions:
        graph_def.ParseFromString(model_file.read())
        _ = tf.import_graph_def(graph_def, name="")

def extract_img_features(img_paths):
    """
    - Runs every image in "img_paths" through the pretrained CNN and
    returns their respective feature vectors (the second-to-last layer
    of the CNN).
    """

    no_of_features = 2048
    img_id_2_feature_vector = {}

    # load the Inception-V3 model:
    load_pretrained_CNN()

    with tf.Session() as sess:
        # get the second-to-last layer in the Inception-V3 model (this
        # is what we will use as a feature vector for each image):
        second_to_last_tensor = sess.graph.get_tensor_by_name("pool_3:0")

        for step, img_path in enumerate(img_paths):
            if step % 100 == 0:
                print step

            # read the image and get its corresponding feature vector:
            img_data = gfile.FastGFile(img_path, "rb").read()
            try:
                feature_vector = sess.run(second_to_last_tensor,
                        feed_dict={"DecodeJpeg/contents:0": img_data})
            except:
                print "JPEG error for:"
                print img_id
                print "******************"
            else:
                # # flatten the features to an np.array:
                feature_vector = np.squeeze(feature_vector)

                # get the image id:
                img_name = img_path.split("/")[3]
                img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
                img_id = int(img_id)

                # save the feature vector and the img id:
                img_id_2_feature_vector[img_id] = feature_vector

        return img_id_2_feature_vector

# define where all train imgs are located:
train_img_dir = "coco/images/train/"
# create a list of the paths to all train imgs:
train_img_paths = [train_img_dir + file_name for file_name in\
                   os.listdir(train_img_dir) if ".jpg" in file_name]
# get the feature vectors for all train imgs:
train_img_id_2_feature_vector = extract_img_features(train_img_paths)

# define where all val imgs are located:
val_img_dir = "coco/images/val/"
# create a list of the paths to all val imgs:
val_img_paths = [val_img_dir + file_name for file_name in\
                 os.listdir(val_img_dir) if ".jpg" in file_name]
# get the feature vectors for all val imgs:
val_img_id_2_feature_vector = extract_img_features(val_img_paths)

# define where all test imgs are located:
test_img_dir = "coco/images/test/"
# create a list of the paths to all test imgs:
test_img_paths = [test_img_dir + file_name for file_name in\
                  os.listdir(test_img_dir) if ".jpg" in file_name]
# get the feature vectors for all test imgs:
test_img_id_2_feature_vector = extract_img_features(test_img_paths)

# save all the feature vectors on disk:
cPickle.dump(train_img_id_2_feature_vector,
             open("coco/data/train_img_id_2_feature_vector", "wb"))
cPickle.dump(val_img_id_2_feature_vector,
             open("coco/data/val_img_id_2_feature_vector", "wb"))
cPickle.dump(test_img_id_2_feature_vector,
             open("coco/data/test_img_id_2_feature_vector", "wb"))
