"""
- DOES: contains a bunch of code snippets that have been tested or used at some
  point. Probably nothing interesting to see here.
"""



import cPickle
import os
import numpy as np
import shutil

#captions_dir = "coco/annotations/"
#ids_dir = "coco/features/"

# load the captions from disk:
#test_captions = pickle.load(open(os.path.join(captions_dir, "test_captions")))

# load the test image ids from disk:
#test_img_ids = pickle.load(open(os.path.join(ids_dir, "img_ids_test")))

#img_id = int(test_img_ids[0])
#img_captions = test_captions[img_id]

#print img_captions

#for caption in img_captions:
#    print caption

# load the vocabulary from disk:
#vocabulary = pickle.load(open(os.path.join(captions_dir, "vocabulary")))

#print vocabulary

# load the embeddings matrix from disk:
#embeddings_matrix = pickle.load(open(os.path.join(captions_dir, "embeddings_matrix")))

#print embeddings_matrix

#test_img_ids = cPickle.load(open("coco/data/test_img_ids"))

#val_img_ids = cPickle.load(open("coco/data/val_img_ids"))

#caption_id_2_img_id = cPickle.load(open("coco/data/caption_id_2_img_id"))

#test_caption_id_2_caption = cPickle.load(open("coco/data/test_caption_id_2_caption"))

#train_caption_id_2_caption = cPickle.load(open("coco/data/train_caption_id_2_caption"))

#val_caption_id_2_caption = cPickle.load(open("coco/data/val_caption_id_2_caption"))

#vocabulary = cPickle.load(open("coco/data/vocabulary"))

#embeddings = cPickle.load(open("coco/data/embeddings_matrix"))

#val_img_id_2_feature_vector = cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
#test_img_id_2_feature_vector = cPickle.load(open("coco/data/test_img_id_2_feature_vector"))
#train_img_id_2_feature_vector = cPickle.load(open("coco/data/train_img_id_2_feature_vector"))

#caption_id = 829719
#img_id = caption_id_2_img_id[caption_id]

#print test_img_id_2_feature_vector[img_id]

#batches_of_captions = cPickle.load(open("coco/data/batches_of_captions"))
#print batches_of_captions
#print caption_id_2_img_id[batches_of_captions[127][123]]

# # tokenize all test captions:
# for step, caption_id in enumerate(test_caption_id_2_caption):
#     if step % 1000 == 0:
#         print "test, tokenizing: ", step
#
#     caption = test_caption_id_2_caption[caption_id]
#
#     # tokenize the caption:
#     tokenized_caption = []
#     for word in caption:
#         if word in vocabulary:
#             word_index = vocabulary.index(word)
#         else:
#             word_index = -99
#         tokenized_caption.append(word_index)
#
#     # convert into a numpy array:
#     tokenized_caption = np.array(tokenized_caption)
#     # save:
#     test_caption_id_2_caption[caption_id] = tokenized_caption
#
# # save all the captions to disk:
# cPickle.dump(test_caption_id_2_caption, open(os.path.join("coco/data/",
#         "testing"), "wb"))

# from utilities import detokenize_caption
#
# vocabulary = cPickle.load(open("coco/data/vocabulary"))
#
# caption = detokenize_caption([0, 123, 34, 1, 8473, 2], vocabulary)
# print caption

# val_caption_id_2_caption = cPickle.load(open("coco/data/val_caption_id_2_caption"))
# vocabulary = cPickle.load(open("coco/data/vocabulary"))
#
#  # tokenize all train captions:
# for step, caption_id in enumerate(val_caption_id_2_caption):
#     if step % 1000 == 0:
#         print "train, tokenizing: ", step
#
#     caption = val_caption_id_2_caption[caption_id]
#
#     # tokenize the caption:
#     tokenized_caption = []
#     for word in caption:
#         if word in vocabulary:
#             word_index = vocabulary.index(word)
#         else:
#             word_index = 1
#         tokenized_caption.append(word_index)
#
#     # convert into a numpy array:
#     tokenized_caption = np.array(tokenized_caption)
#     # save:
#     val_caption_id_2_caption[caption_id] = tokenized_caption
# data_dir = "coco/data/"
#
# cPickle.dump(val_caption_id_2_caption, open(os.path.join(data_dir,
#         "val_caption_id_2_caption"), "wb"))

# test = cPickle.load(open("coco/data/train_caption_length_2_no_of_captions"))
#
# for caption_length in test:
#     print caption_length
#     print test[caption_length]

    # initialize the model containing W_img and b_img:

# ################33
# from LSTM_model import LSTM_Config, LSTM_Model
# import numpy as np
# import tensorflow as tf
#
# config = LSTM_Config()
# dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
#             dtype=np.float32)
# model = LSTM_Model(config, dummy_embeddings, mode="demo")
#
# # create the saver:
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     # restore all model variables:
#     params_dir = "coco/data/img_features_attention/transform_params"
#     saver.restore(sess, "%s/model" % params_dir)
#
#     # get the restored W_img and b_img:
#     with tf.variable_scope("img_transform", reuse=True):
#         W_img = tf.get_variable("W_img")
#         b_img = tf.get_variable("b_img")
#
#         W_img = sess.run(W_img)
#         b_img = sess.run(b_img)
#
#         transform_params = {}
#         transform_params["W_img"] = W_img
#         transform_params["b_img"] = b_img
#         cPickle.dump(transform_params, open("coco/data/img_features_attention/transform_params/numpy_params", "wb"))

# from utilities import plot_comparison_curves
#
# plot_comparison_curves(["models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=1", "models/LSTMs_attention/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=1"], "loss", {"param": "batch size", "param_values": [128, 256]})
# plot_comparison_curves(["models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=1", "models/LSTMs_attention/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=1"], "CIDEr", {"param": "batch size", "param_values": [128, 256]})

# from utilities import plot_performance
# loss_per_epoch1 = cPickle.load(open("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500/losses/loss_per_epoch1"))
# loss_per_epoch2 = cPickle.load(open("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500/losses/loss_per_epoch2"))
#
# loss_per_epoch = loss_per_epoch1 + loss_per_epoch2
#
# cPickle.dump(loss_per_epoch, open("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500/losses/loss_per_epoch", "wb"))
#
# metrics_per_epoch1 = cPickle.load(open("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500/eval_results/metrics_per_epoch1"))
# metrics_per_epoch2 = cPickle.load(open("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500/eval_results/metrics_per_epoch2"))
#
# metrics_per_epoch = metrics_per_epoch1 + metrics_per_epoch2
#
# cPickle.dump(metrics_per_epoch, open("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500/eval_results/metrics_per_epoch", "wb"))
#
# plot_performance("models/LSTMs_attention/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1_hidden_dim_att=500")

# from utilities import plot_comparison_curves
# #
# plot_comparison_curves(["models/GRUs/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1",
#         "models/GRUs/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=2",
#         "models/GRUs/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=3"],
#         "Bleu_4", {"param": "layers", "param_values": ["1", "2", "3"]})

# with open("log.txt") as file:
#     loss_per_epoch = []
#     metrics_per_epoch = []
#
#     for line in file:
#         # remove the new line char at the end:
#         line = line.strip()
#
#         # seperate the word from the word vector:
#         line_elements = line.split(" ")
#         if "CIDEr:" in line_elements:
#             loss = float(line_elements[4])
#             Bleu_4 = float(line_elements[7])
#             CIDEr = float(line_elements[12])
#
#             loss_per_epoch.append(loss)
#
#             epoch_metrics = {}
#             epoch_metrics["CIDEr"] = CIDEr
#             epoch_metrics["Bleu_4"] = Bleu_4
#             epoch_metrics["ROUGE_L"] = 0
#             epoch_metrics["METEOR"] = 0
#
#             metrics_per_epoch.append(epoch_metrics)
#
#     cPickle.dump(loss_per_epoch, open("loss_per_epoch", "wb"))
#     cPickle.dump(metrics_per_epoch, open("metrics_per_epoch", "wb"))

# import cPickle
# import os
# import numpy as np
# import shutil
#
# val_img_dir = "/mnt/train2014/"
# # create a list of the paths to all val imgs:
# val_img_paths = [val_img_dir + file_name for file_name in\
#                  os.listdir(val_img_dir) if ".jpg" in file_name]
#
# for img_path in val_img_paths:
#     img_name = img_path.split("/")[3]
#
#     shutil.move(img_path, "/mnt/imgs/%s" % img_name)

# val_img_ids = cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/val_img_ids"))
# test_img_ids = cPickle.load(open("/home/fregu856/CS224n/project/CS224n_project/coco/data/test_img_ids"))
#
# val_img_id_2_feature_array = {}
# test_img_id_2_feature_array = {}
#
# dir = "coco/data/img_features_attention/"
# # create a list of the paths to all val imgs:
# img_paths = [dir + file_name for file_name in\
#             os.listdir(dir)]
#
# for step, path in enumerate(img_paths):
#     if step % 100 == 0:
#         print step
#
#     file_name = path.split("/")[3]
#     if file_name not in ["transform_params", "-1"]:
#         img_id = int(file_name)
#         feature_array = cPickle.load(open(path))
#         if img_id in val_img_ids:
#             val_img_id_2_feature_array[img_id] = feature_array
#         else:
#             test_img_id_2_feature_array[img_id] = feature_array
#
# cPickle.dump(val_img_id_2_feature_array,
#         open("coco/data/val_img_id_2_feature_array", "wb"))
# cPickle.dump(test_img_id_2_feature_array,
#         open("coco/data/test_img_id_2_feature_array", "wb"))

# from utilities import evaluate_base_model
# evaluate_base_model()

# from utilities import compare_captions
# compare_captions("models/LSTMs/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1", 27, 563)

# caption_id_2_img_id = cPickle.load(open("coco/data/caption_id_2_img_id"))
#
# train_caption_id_2_caption = cPickle.load(open("coco/data/test_caption_id_2_caption"))
# unique_words = []
# img_ids = []
# no_of_captions = 0
# for caption_id in train_caption_id_2_caption:
#     img_id = caption_id_2_img_id[caption_id]
#     if img_id not in img_ids:
#         img_ids.append(img_id)
#         caption = train_caption_id_2_caption[caption_id]
#         no_of_captions += 1
#         caption.pop()
#         caption.pop(0)
#         for word in caption:
#             if word not in unique_words:
#                 unique_words.append(word)
#
# vocabulary = len(unique_words)
# print vocabulary

import cPickle
import random
import numpy as np
import tensorflow as tf
import skimage.io as io
import skimage
import matplotlib.pyplot as plt

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/home/fregu856/CS224n/project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model
from GRU_attention_model import GRU_attention_Config, GRU_attention_Model
from LSTM_attention_model import LSTM_attention_Config, LSTM_attention_Model
from extract_img_features_attention import extract_img_features_attention

# load all needed data:
test_img_ids = cPickle.load(open("coco/data/test_img_ids"))
test_img_id_2_feature_vector =\
            cPickle.load(open("coco/data/test_img_id_2_feature_vector"))
vocabulary = cPickle.load(open("coco/data/vocabulary"))

true_captions_file = "coco/annotations/captions_val2014.json"
coco = COCO(true_captions_file)

demo_img_ids = cPickle.load(open("coco/data/demo_img_ids"))

# initialize the model:
# config = LSTM_Config()
# dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
#             dtype=np.float32)
# model = LSTM_Model(config, dummy_embeddings, mode="demo")
config = LSTM_attention_Config()
dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
            dtype=np.float32)
model = LSTM_attention_Model(config, dummy_embeddings, mode="demo")

# create the saver:
saver = tf.train.Saver()

with tf.Session() as sess:
    # restore the best model:
    #saver.restore(sess, "models/LSTMs/best_model/model")
    saver.restore(sess, "models/LSTMs_attention/best_model/model")

    img_number = 1
    for img_id in demo_img_ids:
        print img_number
        # get the img's file name:
        img_id = int(img_id)
        img = coco.loadImgs(img_id)[0]
        img_file_name = img["file_name"]

        # get the img's features:
        #img_features = test_img_id_2_feature_vector[img_id]
        img_features = cPickle.load(
                    open("coco/data/img_features_attention/%d" % img_id))

        #img_caption = model.generate_img_caption(sess, img_features, vocabulary)
        img_caption, attention_maps = model.generate_img_caption(sess,
                       img_features, vocabulary)

        # display the img and its generated caption:
        I = io.imread("coco/images/test/%s" % img_file_name)
        plt.figure()
        plt.imshow(I)
        plt.axis('off')
        plt.title(img_caption, fontsize=15)
        plt.savefig("coco/captioned_imgs/attention_%d" % img_number, bbox_inches="tight")

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
            plt.savefig("coco/captioned_imgs/map_%d" % img_number, bbox_inches="tight")

        plt.close()
        img_number += 1


# dir = "coco/data/img_features_attention/"
# # create a list of the paths to all val imgs:
# paths = [dir + file_name for file_name in os.listdir(dir)]
#
# img_id_2_feature_array = {}
# cPickle.dump(img_id_2_feature_array,
#             open("/mnt/img_id_2_feature_array", "wb"))
# cPickle.dump(img_id_2_feature_array,
#             open("/mnt/img_id_2_feature_array", "wb"))
#
# for step, path in enumerate(paths):
#     if step % 1000 == 0:
#         print step
#         log(str(step))
#     img_id = int(path.split("/")[3])
#     feature_array = cPickle.load(open(path))
#     img_id_2_feature_array[img_id] = feature_array
#
# cPickle.dump(img_id_2_feature_array,
#              open("/mnt/img_id_2_feature_array", "wb"))
