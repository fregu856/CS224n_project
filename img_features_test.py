import os 
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pyplot as plt
import pickle

name = "_train_1"

# define where the pretrained inception model is located:
model_dir = "inception"

# define where all images are located:
img_dir = "coco/images/"
# create a list of the paths to all images:
img_paths = [img_dir + file_name for file_name in\
        os.listdir(img_dir) if ".jpg" in file_name]
        
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
    of the CNN) in "img_feature_vectors".
    """
    
    no_of_features = 2048
    no_of_imgs = len(img_paths)
    img_feature_vectors = np.empty((no_of_imgs, no_of_features))
    img_ids = []
    
    # load the Inception-V3 model:
    load_pretrained_CNN()
    
    with tf.Session() as sess:
        # get the second-to-last layer in the Inception-V3 model (this
        # is what we will use as a feature vector for each image):
        second_to_last_tensor = sess.graph.get_tensor_by_name(
                "pool_3:0")
        
        for img_index, img_path in enumerate(img_paths):
            print "Processing %s" % img_path
            
            if not gfile.Exists(img_path):
                tf.logging.fatal("File does not exist:", img_path)
                
            # read the image and get its corresponding feature vector:
            img_data = gfile.FastGFile(img_path, "rb").read()
            feature_vector = sess.run(second_to_last_tensor, 
                    feed_dict={"DecodeJpeg/contents:0": img_data})
            # # flatten the features to an np.array:
            feature_vector = np.squeeze(feature_vector)
            # # save the image features:
            img_feature_vectors[img_index, :] = feature_vector
            
            # save the image id:
            img_name = img_path.split("/")[2]
            img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
            img_ids.append(img_id)
            
        return img_feature_vectors, img_ids
        
img_feature_vectors, img_ids = extract_img_features(img_paths)

print img_feature_vectors
print img_ids

# save the feature vectors and names on disk:
pickle.dump(img_feature_vectors, 
        open(os.path.join(img_dir, "img_feature_vectors" + name), "wb"))
pickle.dump(img_ids, 
        open(os.path.join(img_dir, "img_ids" + name), "wb"))

# load the feature vectors and names from disk:
#features = pickle.load(open(os.path.join(img_dir, "img_feature_vectors")))
#names = pickle.load(open(os.path.join(img_dir, "img_names")))
#print features
#print names