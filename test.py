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
# loss_per_epoch1 = cPickle.load(open("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3/losses/loss_per_epoch1"))
# loss_per_epoch2 = cPickle.load(open("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3/losses/loss_per_epoch2"))
#
# loss_per_epoch = loss_per_epoch1 + loss_per_epoch2
#
# cPickle.dump(loss_per_epoch, open("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3/losses/loss_per_epoch", "wb"))
#
# metrics_per_epoch1 = cPickle.load(open("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3/eval_results/metrics_per_epoch1"))
# metrics_per_epoch2 = cPickle.load(open("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3/eval_results/metrics_per_epoch2"))
#
# metrics_per_epoch = metrics_per_epoch1 + metrics_per_epoch2
#
# cPickle.dump(metrics_per_epoch, open("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3/eval_results/metrics_per_epoch", "wb"))
#
# plot_performance("models/LSTMs/model_keep=0.50_batch=256_hidden_dim=200_embed_dim=300_layers=3")

from utilities import plot_comparison_curves

plot_comparison_curves(["models/LSTMs/model_keep=0.75_batch=128_hidden_dim=400_embed_dim=300_layers=1",
        "models/LSTMs/model_keep=0.75_batch=256_hidden_dim=400_embed_dim=300_layers=1",
        "models/LSTMs/model_keep=0.75_batch=512_hidden_dim=400_embed_dim=300_layers=1"],
        "Bleu_4", {"param": "batch size", "param_values": [128, 256, 512]})

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

import cPickle
import os
import numpy as np
import shutil

val_img_dir = "/mnt/train2014/"
# create a list of the paths to all val imgs:
val_img_paths = [val_img_dir + file_name for file_name in\
                 os.listdir(val_img_dir) if ".jpg" in file_name]

for img_path in val_img_paths:
    img_name = img_path.split("/")[3]

    shutil.move(img_path, "/mnt/imgs/%s" % img_name)
