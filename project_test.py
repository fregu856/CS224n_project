import os 
import re

import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.platform import gfile
import numpy as np
import matplotlib.pyplot as plt
import pickle

# define where the pretrained inception model is located:
model_dir = "inception"

# define where all images are located:
images_dir = "images/"
# create a list of the paths to all images:
image_paths = [images_dir + file_name for file_name in\
        os.listdir(images_dir) if ".jpg" in file_name]
        
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
        
def extract_image_features(image_paths):
    """
    - Runs every image in "image_paths" through the pretrained CNN and
    returns their respective feature vectors (the second-to-last layer 
    of the CNN) in "image_feature_vectors".
    """
    
    no_of_features = 2048
    no_of_images = len(image_paths)
    image_feature_vectors = np.empty((no_of_images, no_of_features))
    image_names = []
    
    # load the Inception-V3 model:
    load_pretrained_CNN()
    
    with tf.Session() as sess:
        # get the second-to-last layer in the Inception-V3 model (this
        # is what we will use as a feature vector for each image):
        second_to_last_tensor = sess.graph.get_tensor_by_name(
                "pool_3:0")
        
        for image_index, image_path in enumerate(image_paths):
            print "Processing %s" % image_path
            
            if not gfile.Exists(image_path):
                tf.logging.fatal("File does not exist:", image_path)
                
            # read the image and get its corresponding feature vector:
            image_data = gfile.FastGFile(image_path, "rb").read()
            feature_vector = sess.run(second_to_last_tensor, 
                    feed_dict={"DecodeJpeg/contents:0": image_data})
            # # flatten the features to an np.array:
            feature_vector = np.squeeze(feature_vector)
            # # save the image features:
            image_feature_vectors[image_index, :] = feature_vector
            
            # save the image name:
            image_name = image_path.split("/")[1]
            image_names.append(image_name)
            
        return image_feature_vectors, image_names
        
image_feature_vectors, image_names = extract_image_features(image_paths)

print image_feature_vectors
print image_names

    
        


