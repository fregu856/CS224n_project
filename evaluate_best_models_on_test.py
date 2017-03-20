"""
- ASSUMES: that preprocess_captions.py, extract_img_features.py and
  extract_img_features_attention.py has already been run. That the weights for the
  best LSTM/GRU/LSTM_attention/GRU_attention model has been placed in
  models/**model_type**/best_model with names model.filetype.

- DOES: generates captions for all 5000 imgs in test using the best
  LSTM/GRU/LSTM_attention/GRU_attention model, evaluates the captions and
  returns the metric scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, METEOR and
  ROUGE_L).
"""

import numpy as np
import tensorflow as tf
import cPickle
import json

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model
from GRU_attention_model import GRU_attention_Config, GRU_attention_Model
from LSTM_attention_model import LSTM_attention_Config, LSTM_attention_Model
from utilities import evaluate_captions, detokenize_caption

def evaluate_best_model(model_type, test_set, vocabulary, train_captions):
    # initialize the model:
    if model_type == "GRU":
        config = GRU_Config()
        dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                    dtype=np.float32)
        model = GRU_Model(config, dummy_embeddings, mode="demo")
    elif model_type == "LSTM":
        config = LSTM_Config()
        dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                    dtype=np.float32)
        model = LSTM_Model(config, dummy_embeddings, mode="demo")
    elif model_type == "LSTM_attention":
        config = LSTM_attention_Config()
        dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                    dtype=np.float32)
        model = LSTM_attention_Model(config, dummy_embeddings, mode="demo")
    elif model_type == "GRU_attention":
        config = GRU_attention_Config()
        dummy_embeddings = np.zeros((config.vocab_size, config.embed_dim),
                    dtype=np.float32)
        model = GRU_attention_Model(config, dummy_embeddings, mode="demo")

    # create the saver:
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # restore the best model:
        if model_type == "GRU":
            saver.restore(sess, "models/GRUs/best_model/model")
        elif model_type == "LSTM":
            saver.restore(sess, "models/LSTMs/best_model/model")
        elif model_type == "LSTM_attention":
            saver.restore(sess, "models/LSTMs_attention/best_model/model")
        elif model_type == "GRU_attention":
            saver.restore(sess, "models/GRUs_attention/best_model/model")

        captions = []
        no_of_new_captions = 0
        no_of_old_captions = 0
        unique_words = []
        for step, (img_id, img_vector) in enumerate(test_set):
            if step % 100 == 0:
                print "generating captions on test: %d" % step

            # generate a caption for the img:
            if model_type in ["LSTM", "GRU"]:
                img_caption = model.generate_img_caption(sess, img_vector, vocabulary)
            elif model_type in ["LSTM_attention", "GRU_attention"]:
                # get the img features from disk:
                img_features = cPickle.load(
                            open("coco/data/img_features_attention/%d" % img_id))
                # generate a caption:
                img_caption, attention_maps = model.generate_img_caption(sess,
                            img_features, vocabulary)

            # save the generated caption together with the img id in the format
            # expected by the COCO evaluation script:
            caption_obj = {}
            caption_obj["image_id"] = img_id
            caption_obj["caption"] = img_caption
            captions.append(caption_obj)

            # check if the generated caption is new or is in train:
            if img_caption in train_captions:
                no_of_old_captions += 1
            else:
                no_of_new_captions += 1

            # check if there are any words in the caption that he model hasn't
            # generated before:
            for word in img_caption.split(" "):
                if word not in unique_words:
                    unique_words.append(word)

    # save the captions as a json file (will be used by the eval script):
    captions_file = "coco/data/test_captions.json"
    with open(captions_file, "w") as file:
        json.dump(captions, file, sort_keys=True, indent=4)

    # evaluate the generated captions:
    results_dict = evaluate_captions(captions_file)

    # compute the ratio of new captions:
    new_captions_ratio = float(no_of_new_captions)/float(no_of_new_captions +
                no_of_old_captions)

    # get the number of unique words that the model generated:
    vocab_size = len(unique_words)

    results_dict["new_captions_ratio"] = new_captions_ratio
    results_dict["vocab_size"] = vocab_size

    return results_dict

def main():
    # load the vocabulary:
    vocabulary = cPickle.load(open("coco/data/vocabulary"))
    # get the map from img id to feature vector:
    test_img_id_2_feature_vector =\
                cPickle.load(open("coco/data/test_img_id_2_feature_vector"))
    # turn the map into a list of tuples (to make it iterable):
    test_set = test_img_id_2_feature_vector.items()

    # get all captions in the training set:
    train_caption_id_2_caption = cPickle.load(open("coco/data/train_caption_id_2_caption"))
    train_captions = []
    for caption_id in train_caption_id_2_caption:
        caption = train_caption_id_2_caption[caption_id]
        caption = detokenize_caption(caption, vocabulary)
        train_captions.append(caption)

    # evaluate best LSTM model:
    LSTM_results_dict = evaluate_best_model("LSTM", test_set, vocabulary,
               train_captions)
    # evaluate best GRU model:
    GRU_results_dict = evaluate_best_model("GRU", test_set, vocabulary,
                train_captions)
    # evaluate best LSTM attention model:
    LSTM_att_results_dict =\
                evaluate_best_model("LSTM_attention", test_set, vocabulary,
                train_captions)
    # evaluate best GRU attention model:
    GRU_att_results_dict =\
             evaluate_best_model("GRU_attention", test_set, vocabulary,
                train_captions)

    # put all results in one dict:
    results = {}
    results["LSTM"] = LSTM_results_dict
    results["LSTM_attention"] = LSTM_att_results_dict
    results["GRU"] = GRU_results_dict
    results["GRU_attention"] = GRU_att_results_dict

    # print all results
    print results

    # save all results to disk:
    cPickle.dump(results,
            open("coco/data/eval_results_on_test", "wb"))

if __name__ == '__main__':
    main()
