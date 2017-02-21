import cPickle
import numpy as np
import os
import random

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/home/fregu856/CS224n/project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

# add the "coco-caption" dir to the path so that "pycocoevalcap" can be found:
sys.path.append("/home/fregu856/CS224n/project/CS224n_project/coco/coco-caption")
from pycocoevalcap.eval import COCOEvalCap

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

def get_batches(model_obj):
    batch_size = model_obj.config.batch_size

    # group all caption ids in batches:
    batches_of_caption_ids = []
    for caption_length in model_obj.caption_length_2_no_of_captions:
        caption_ids = model_obj.caption_length_2_caption_ids[caption_length]
        # randomly shuffle the order of the caption ids:
        random.shuffle(caption_ids)
        no_of_captions = model_obj.caption_length_2_no_of_captions[caption_length]
        no_of_full_batches = int(no_of_captions/batch_size)

        # add all full batches to batches_of_caption_ids:
        for i in range(no_of_full_batches):
            batch_caption_ids = caption_ids[i*batch_size:(i+1)*batch_size]
            batches_of_caption_ids.append(batch_caption_ids)

        # get the remaining caption ids and add to batches_of_captions (not a
        # full batch, i.e, it will contain fewer than "batch_size" captions):
        #batch_caption_ids = caption_ids[no_of_full_batches*batch_size:]
        #batches_of_caption_ids.append(batch_caption_ids)

    # randomly shuffle the order of the batches:
    random.shuffle(batches_of_caption_ids)

    return batches_of_caption_ids

def get_batch_ph_data(model_obj, batch_caption_ids):
    # get the dimension parameters:
    batch_size = model_obj.config.batch_size
    img_dim = model_obj.config.img_dim
    caption_length = len(model_obj.train_caption_id_2_caption[batch_caption_ids[0]])

    captions = np.zeros((batch_size, caption_length))
    # (row i of captions will be the tokenized caption for ex i in the batch)
    img_vectors = np.zeros((batch_size, img_dim))
    # (row i of img_vectors will be the img feature vector for ex i in the batch)
    labels = -np.ones((batch_size, caption_length + 1))
    # (row i of labels will be the targets for ex i in the batch)

    # pupulate the return data:
    for i in range(len(batch_caption_ids)):
        caption_id = batch_caption_ids[i]
        img_id = model_obj.caption_id_2_img_id[caption_id]
        img_vector = model_obj.train_img_id_2_feature_vector[img_id]
        caption = model_obj.train_caption_id_2_caption[caption_id]

        captions[i] = caption
        img_vectors[i] = img_vector
        labels[i, 1:caption_length] = caption[1:]

        # example to explain labels:
        # caption == [<SOS>, a, cat, <EOS>]
        # caption_length == 4
        # labels[i] == [-1, -1, -1, -1, -1]
        # caption[1:] == [a, cat, <EOS>]
        # labels[i, 1:caption_length] = caption[1:] gives:
        # labels[i] == [-1, a, cat, <EOS>, -1]
        # corresponds to the input:
        # img, <SOS>, a, cat, <EOS>
        # img: no prediciton should be made (-1)
        # <SOS>: should predict a (a)
        # a: should predict cat (cat)
        # cat: should predict <EOS> (<EOS>)
        # <EOS>: no prediction should be made (-1)

    return captions, img_vectors, labels

def train_data_iterator(model_obj):
    # get the batches of caption ids:
    batches_of_caption_ids = get_batches(model_obj)

    for batch_of_caption_ids in batches_of_caption_ids:
        # get the batch's data in a format ready to be fed into the placeholders:
        captions, img_vectors, labels = get_batch_ph_data(model_obj,
                    batch_of_caption_ids)

        # yield the data to enable iteration (will be able to do:
        # for captions, img_vector, labels in train_data_iterator(config):)
        yield (captions, img_vectors, labels)

def detokenize_caption(tokenized_caption, vocabulary):
    caption_vector = []
    for word_index in tokenized_caption:
        word = vocabulary[word_index]
        caption_vector.append(word)

    # remove <SOS> and <EOS>:
    caption_vector.pop(0)
    caption_vector.pop()

    # turn the caption vector into a string:
    caption = " ".join(caption_vector)

    return caption

def evaluate_captions(captions_file):
    # define where the ground truth captions for the val imgs are located:
    true_captions_file = "coco/annotations/captions_val2014.json"

    coco = COCO(true_captions_file)
    cocoRes = coco.loadRes(captions_file)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # set the imgs to be evaluated to the ones we have generated captions for:
    cocoEval.params["image_id"] = cocoRes.getImgIds()
    # evaluate the captions (compute metrics):
    cocoEval.evaluate()
    # get the dict containing all computed metrics and metric scores:
    results_dict = cocoEval.eval

    return results_dict
