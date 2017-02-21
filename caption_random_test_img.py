# NOTE! val should be replaced by test once the project is finished!

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

from model import Config, Model

val_img_ids = cPickle.load(open("coco/data/val_img_ids"))
val_img_id_2_feature_vector = cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
vocabulary = cPickle.load(open("coco/data/vocabulary"))

random.shuffle(val_img_ids)
img_id = int(val_img_ids[0])


feature_vector = val_img_id_2_feature_vector[img_id]

config = Config()
GloVe_embeddings = cPickle.load(open("coco/data/embeddings_matrix"))
GloVe_embeddings = GloVe_embeddings.astype(np.float32)
model = Model(config, GloVe_embeddings, mode="demo")

saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, "models/LSTMs/best_model/model")
    img_caption = model.generate_img_caption(sess, feature_vector, vocabulary)

imgId = img_id

true_captions_file = "coco/annotations/captions_val2014.json"
coco = COCO(true_captions_file)

img = coco.loadImgs(imgId)[0]
I = io.imread("coco/images/val/%s" % img["file_name"])
plt.imshow(I)
plt.axis('off')
plt.title(img_caption)
plt.show()
