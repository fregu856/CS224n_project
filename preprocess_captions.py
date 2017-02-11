"""
- Assumes that "split_img_ids.py" already has been run.
"""

import numpy as np
import cPickle
import os
import re

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/home/fregu856/CS224n/project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

captions_dir = "coco/annotations/"
data_dir = "coco/data/"

# load the test img ids from disk:
test_img_ids = cPickle.load(open("coco/data/test_img_ids"))
#load the val img ids from disk>
val_img_ids = cPickle.load(open("coco/data/val_img_ids"))

def get_captions(type_of_data):
    captions_file = "coco/annotations/captions_%s2014.json" % type_of_data

    # initialize COCO api for captions:
    coco = COCO(captions_file)

    # get indices for all "type_of_data" images (all train or val images) (original split on mscoco.org):
    img_ids = coco.getImgIds()

    for step, img_id in enumerate(img_ids):
        if step % 1000 == 0:
            print step

        # get the ids of all captions for the image:
        caption_ids = coco.getAnnIds(imgIds=img_id)
        # get all caption objects for the image:
        caption_objs = coco.loadAnns(caption_ids)

        for caption_obj in caption_objs:
            # save the caption id and the corresponding img id:
            caption_id = caption_obj["id"]
            caption_id_2_img_id[caption_id] = img_id

            # get the caption:
            caption = caption_obj["caption"]
            # remove empty spaces in the start or end of the caption:
            caption = caption.strip()
            # make the caption lower case:
            caption = caption.lower()
            # remove all non-alphanum chars (keep spaces between word):
            caption = re.sub("[^a-z0-9 ]+", "", caption)
            # remove all double spaces in the caption:
            caption = re.sub("  ", " ", caption)
            # convert the caption into a vector of words:
            caption = caption.split(" ")
            # remove any empty chars still left in the caption:
            while "" in caption:
                index = caption.index("")
                del caption[index]

            # store the caption in the corresponding captions dict:
            if img_id in test_img_ids:
                test_caption_id_2_caption[caption_id] = caption
            elif img_id in val_img_ids:
                val_caption_id_2_caption[caption_id] = caption
            else:
                train_caption_id_2_caption[caption_id] = caption

train_caption_id_2_caption = {}
test_caption_id_2_caption = {}
val_caption_id_2_caption = {}
caption_id_2_img_id = {}
get_captions("train")
get_captions("val")

# save the caption_id to img_id mapping dict to disk:
cPickle.dump(caption_id_2_img_id,
        open(os.path.join(data_dir, "caption_id_2_img_id"), "wb"))

# get all words that have a pretrained word embedding:
pretrained_words = []
with open(os.path.join(captions_dir, "glove.6B.300d.txt")) as file:
    for line in file:
        line_elements = line.split(" ")
        word = line_elements[0]
        pretrained_words.append(word)

# count how many times each word occur in the training set:
word_counts = {}
for caption_id in train_caption_id_2_caption:
    caption = train_caption_id_2_caption[caption_id]
    for word in caption:
        if word not in word_counts:
            word_counts[word] = 1
        else:
            word_counts[word] += 1

# create a vocabulary of all words that appear 5 or more times in the
# training set AND have a pretrained word embedding:
vocabulary = []
for word in word_counts:
    word_count = word_counts[word]
    if word_count >= 5 and word in pretrained_words:
        vocabulary.append(word)

# replace all words in train that are not in the vocabulary with an
# <UNK> token AND prepend each caption with an <SOS> token AND append
# each caption with an <EOS> token:
for step, caption_id in enumerate(train_caption_id_2_caption):
    if step % 1000 == 0:
        print "train: ", step

    caption = train_caption_id_2_caption[caption_id]
    for word_index in range(len(caption)):
        word = caption[word_index]
        if word not in vocabulary:
            caption[word_index] = "<UNK>"
    # prepend the caption with an <SOS> token;
    caption.insert(0, "<SOS>")
    # append the caption with an <EOS> token:
    caption.append("<EOS>")

# add "<SOS>", "<UNK>" and "<EOS>" to the vocabulary:
vocabulary.insert(0, "<EOS>")
vocabulary.insert(0, "<UNK>")
vocabulary.insert(0, "<SOS>")

# save the vocabulary to disk:
cPickle.dump(vocabulary,
        open(os.path.join(data_dir, "vocabulary"), "wb"))

# prepend each caption in val with an <SOS> token AND append each
# caption with an <EOS> token:
for step, caption_id in enumerate(val_caption_id_2_caption):
    if step % 1000 == 0:
        print "val: ", step

    caption = val_caption_id_2_caption[caption_id]
    # prepend the caption with an <SOS> token;
    caption.insert(0, "<SOS>")
    # append tge caption with an <EOS> token:
    caption.append("<EOS>")

# prepend each caption in test with an <SOS> token AND append each
# caption with an <EOS> token:
for step, caption_id in enumerate(test_caption_id_2_caption):
    if step % 1000 == 0:
        print "test: ", step

    caption = test_caption_id_2_caption[caption_id]
    # prepend the caption with an <SOS> token;
    caption.insert(0, "<SOS>")
    # append tge caption with an <EOS> token:
    caption.append("<EOS>")

# tokenize all train captions:
for step, caption_id in enumerate(train_caption_id_2_caption):
    if step % 1000 == 0:
        print "train, tokenizing: ", step

    caption = train_caption_id_2_caption[caption_id]

    # tokenize the caption:
    tokenized_caption = []
    for word in caption:
        word_index = vocabulary.index(word)
        tokenized_caption.append(word_index)

    # convert into a numpy array:
    tokenized_caption = np.array(tokenized_caption)
    # save:
    train_caption_id_2_caption[caption_id] = tokenized_caption

# save all the captions to disk:
cPickle.dump(train_caption_id_2_caption, open(os.path.join(data_dir,
        "train_caption_id_2_caption"), "wb"))
cPickle.dump(test_caption_id_2_caption, open(os.path.join(data_dir,
        "test_caption_id_2_caption"), "wb"))
cPickle.dump(val_caption_id_2_caption, open(os.path.join(data_dir,
        "val_caption_id_2_caption"), "wb"))
