import numpy as np
import tensorflow as tf
import cPickle
import json

from GRU_model import GRU_Config, GRU_Model
from LSTM_model import LSTM_Config, LSTM_Model
from GRU_attention_model import GRU_attention_Config, GRU_attention_Model
from LSTM_attention_model import LSTM_attention_Config, LSTM_attention_Model
from utilities import evaluate_captions

def evaluate_best_model(model_type, test_set, vocabulary):
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

    # save the captions as a json file (will be used by the eval script):
    captions_file = "coco/data/test_captions.json"
    with open(captions_file, "w") as file:
        json.dump(captions, file, sort_keys=True, indent=4)

    # evaluate the generated captions:
    results_dict = evaluate_captions(captions_file)

    return results_dict

def main():
    # load the vocabulary:
    vocabulary = cPickle.load(open("coco/data/vocabulary"))
    # get the map from img id to feature vector:
    test_img_id_2_feature_vector =\
                cPickle.load(open("coco/data/test_img_id_2_feature_vector"))
    # turn the map into a list of tuples (to make it iterable):
    test_set = test_img_id_2_feature_vector.items()

    # evaluate best LSTM model:
    LSTM_results_dict = evaluate_best_model("LSTM", test_set, vocabulary)
    # evaluate best GRU model:
    GRU_results_dict = evaluate_best_model("GRU", test_set, vocabulary)
    # evaluate best LSTM attention model:
    LSTM_att_results_dict =\
                evaluate_best_model("LSTM_attention", test_set, vocabulary)
    # evaluate best GRU attention model:
    GRU_att_results_dict =\
                evaluate_best_model("GRU_attention", test_set, vocabulary)

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
