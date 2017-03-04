from flask import Flask, render_template, request

import cPickle
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage
import matplotlib.pyplot as plt

import sys
sys.path.append("/home/fregu856/CS224n/project/CS224n_project")
sys.path.append("/home/fregu856/CS224n/project/CS224n_project/coco/PythonAPI")

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model
from GRU_attention_model import GRU_attention_Config, GRU_attention_Model
from LSTM_attention_model import LSTM_attention_Config, LSTM_attention_Model
from extract_img_features_attention import extract_img_features_attention
from pycocotools.coco import COCO

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    try:
        # pick a random test img if no img id was specified:
        random.shuffle(val_img_ids)
        img_id = int(val_img_ids[0])

        # get the img's file name:
        img = coco.loadImgs(img_id)[0]
        img_file_name = img["file_name"]

        # get the img's features:
        img_features = val_img_id_2_feature_vector[img_id]

        # caption the img (using the best model):
        img_caption = model.generate_img_caption(sess, img_features, vocabulary)

        # display the img and its generated caption:
        I = io.imread("/home/fregu856/CS224n/project/CS224n_project/coco/images/val/%s" % img_file_name)
        plt.imshow(I)
        plt.axis('off')
        plt.title(img_caption, fontsize=15)

        plt.savefig("static/images/captioned_img.jpg")

        return render_template("index.html")
    except Exception as e:
        return render_template("500.html", error = str(e))

if __name__ == '__main__':
    val_img_ids = cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/val_img_ids"))
    val_img_id_2_feature_vector =\
                cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/val_img_id_2_feature_vector"))
    vocabulary = cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/vocabulary"))

    true_captions_file = "/home/fregu856/CS224n/project/CS224n_project/coco/annotations/captions_val2014.json"
    coco = COCO(true_captions_file)

    config = LSTM_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_Model(config, dummy_embeddings, mode="demo")

    # create the saver:
    saver = tf.train.Saver()

    sess = tf.Session()
    # restore the best model:
    saver.restore(sess, "/home/fregu856/CS224n/project/CS224n_project/models/LSTMs/best_model/model")

    app.run()
