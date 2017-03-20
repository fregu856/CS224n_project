"""
- DOES: contains backend code for local live demo webpage.
"""

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
        if request.method == "POST":
            # pick a random test img:
            random.shuffle(test_img_ids)
            img_id = int(test_img_ids[0])

            # get the img's file name:
            img = coco.loadImgs(img_id)[0]
            img_file_name = img["file_name"]

            # get the model type:
            model_type = request.form["button"]

            if model_type == "no attention":
                # get the img's features:
                img_features = test_img_id_2_feature_vector[img_id]

                # caption the img (using the best model):
                img_caption = model.generate_img_caption(sess, img_features, vocabulary)

                # save the img and its generated caption:
                I = io.imread("/home/fregu856/CS224n/project/CS224n_project/coco/images/test/%s" % img_file_name)
                plt.figure(1)
                plt.imshow(I)
                plt.axis('off')
                plt.title(img_caption, fontsize=15)
                plt.savefig("static/images/captioned_img_no_att.jpg", bbox_inches="tight")

            elif model_type == "attention":
                # get the img's features:
                #extract_img_features_attention(["/home/fregu856/CS224n/project/CS224n_project/coco/images/val/%s" % img_file_name], demo=True)
                #img_features = cPickle.load(
                #            open("/home/fregu856/CS224n/project/CS224n_project/coco/data/img_features_attention/%d" % -1))
                img_features = test_img_id_2_feature_array[img_id]

                # caption the img (using the best model):
                img_caption, attention_maps = model_att.generate_img_caption(sess, img_features, vocabulary)

                # save the img and its generated caption:
                I = io.imread("/home/fregu856/CS224n/project/CS224n_project/coco/images/test/%s" % img_file_name)
                plt.figure()
                plt.imshow(I)
                plt.axis('off')
                plt.title(img_caption, fontsize=15)
                plt.savefig("static/images/captioned_img_att.jpg", bbox_inches="tight")

                # get a gray scale version of the img:
                I_gray = skimage.color.rgb2gray(I)
                # get some img paramaters:
                height, width = I_gray.shape
                height_block = int(height/8.)
                width_block = int(width/8.)
                # turn the caption into a vector of the words:
                img_caption_vector = img_caption.split(" ")
                caption_length = len(img_caption_vector)

                plt.figure(figsize=(8, 8))

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
                    plt.title(word, fontsize=15)

                plt.savefig("static/images/attention_map.jpg", bbox_inches="tight")

            elif model_type == "both":
                # get the img's features:
                img_features = test_img_id_2_feature_vector[img_id]

                # caption the img (using the best model):
                img_caption = model.generate_img_caption(sess, img_features, vocabulary)

                # save the img and its generated caption:
                I = io.imread("/home/fregu856/CS224n/project/CS224n_project/coco/images/test/%s" % img_file_name)
                plt.figure()
                plt.imshow(I)
                plt.axis('off')
                plt.title(img_caption, fontsize=15)
                plt.savefig("static/images/captioned_img_no_att.jpg", bbox_inches="tight")

                # get the img's features:
                #extract_img_features_attention(["/home/fregu856/CS224n/project/CS224n_project/coco/images/val/%s" % img_file_name], demo=True)
                #img_features = cPickle.load(
                #            open("/home/fregu856/CS224n/project/CS224n_project/coco/data/img_features_attention/%d" % -1))
                img_features = test_img_id_2_feature_array[img_id]

                # caption the img (using the best model):
                img_caption, attention_maps = model_att.generate_img_caption(sess, img_features, vocabulary)

                # save the img and its generated caption:
                plt.figure()
                plt.imshow(I)
                plt.axis('off')
                plt.title(img_caption, fontsize=15)
                plt.savefig("static/images/captioned_img_att.jpg", bbox_inches="tight")

                # get a gray scale version of the img:
                I_gray = skimage.color.rgb2gray(I)
                # get some img paramaters:
                height, width = I_gray.shape
                height_block = int(height/8.)
                width_block = int(width/8.)
                # turn the caption into a vector of the words:
                img_caption_vector = img_caption.split(" ")
                caption_length = len(img_caption_vector)

                plt.figure(figsize=(8, 8))

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
                    plt.title(word, fontsize=15)

                plt.savefig("static/images/attention_map.jpg", bbox_inches="tight")

            return render_template("index.html", model_type = model_type)
        else:
            return render_template("index.html", model_type = "no attention")

    except Exception as e:
        return render_template("500.html", error = str(e))

@app.errorhandler(404)
def page_not_found(e):
    try:
        return render_template("404.html")
    except Exception as e:
        return render_template("500.html", error = str(e))

if __name__ == '__main__':
    # load all needed data:
    # (demo_img_ids is just 500 random imgs from test which I have checked are not inappropriate)
    test_img_ids = cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/demo_img_ids"))
    test_img_id_2_feature_vector =\
                cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/test_img_id_2_feature_vector"))
    vocabulary = cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/vocabulary"))
    test_img_id_2_feature_array =\
                cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/test_img_id_2_feature_array"))
    true_captions_file = "/home/fregu856/CS224n/project/CS224n_project/coco/annotations/captions_val2014.json"
    coco = COCO(true_captions_file)

    # initialize the no attention model:
    config = LSTM_Config()
    dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                dtype=np.float32)
    model = LSTM_Model(config, dummy_embeddings, mode="demo")

    # initialize the attention model:
    config_att = LSTM_attention_Config()
    dummy_embeddings = np.zeros((config_att.vocab_size, config_att.embed_dim),
                dtype=np.float32)
    model_att = LSTM_attention_Model(config_att, dummy_embeddings, mode="demo")

    sess = tf.Session()

    # seperate the variables in the attention model from the variables in the no
    # attention model:
    att_vars = []
    no_att_vars = []
    all_vars = tf.global_variables()
    for var in all_vars:
        try:
            # try to restore the variable using the no attention model:
            saver = tf.train.Saver([var])
            saver.restore(sess, "/home/fregu856/CS224n/project/CS224n_project/models/LSTMs/best_model/model")
        except:
            # if it wasn't possible, the variable belongs to the attention model:
            att_vars.append(var)
        else:
            # it if was possible, the variable belongs to the no attention model:
            no_att_vars.append(var)

    # create a saver for the atttention model variables:
    saver_att = tf.train.Saver(att_vars)
    # create a saver for the no atttention model variables:
    saver_no_att = tf.train.Saver(no_att_vars)

    # restore the no attention model:
    saver_no_att.restore(sess, "/home/fregu856/CS224n/project/CS224n_project/models/LSTMs/best_model/model")

    # restore the attention model:
    saver_att.restore(sess, "/home/fregu856/CS224n/project/CS224n_project/models/LSTMs_attention/best_model/model")

    # launch the webpage:
    app.run()

    # close the tensorflow sesson before closing the webpage:
    sess.close()
