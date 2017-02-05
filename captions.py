import numpy as np
import pickle
import os
import re

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/afs/.ir.stanford.edu/users/f/r/fregu856/CS224n/Project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

captions_dir = "coco/annotations/"
ids_dir = "coco/features/"

# load the test image ids from disk:        
test_img_ids = pickle.load(open(os.path.join(ids_dir, "img_ids_test")))

# load the val image ids from disk:        
val_img_ids = pickle.load(open(os.path.join(ids_dir, "img_ids_val")))

def get_captions(type_of_data, train_captions, test_captions, val_captions):
    captions_file = "coco/annotations/captions_%s2014.json" % type_of_data

    # initialize COCO api for captions:
    coco=COCO(captions_file)

    # get indices for all "type_of_data" images (all train or val images) (split on mscoco.org):
    img_ids = coco.getImgIds()

    for step, img_id in enumerate(img_ids):
        if step % 1000 == 0:
            print step
            
        # get the ids of all captions for the image:
        caption_ids = coco.getAnnIds(imgIds=img_id)
        # get all caption objects for the image:
        caption_objs = coco.loadAnns(caption_ids)
        
        # get all captions for the image:
        captions_vec = []
        for caption_obj in caption_objs:
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
            # add the caption to the vector of captions:
            captions_vec.append(caption)
            
        # store the captions in the corresponding captions dict:
        if str(img_id) in test_img_ids:
            test_captions[img_id] = captions_vec
        elif str(img_id) in val_img_ids:
            val_captions[img_id] = captions_vec
        else:
            train_captions[img_id] = captions_vec

train_captions = {}
test_captions = {}
val_captions = {}
get_captions("train", train_captions, test_captions, val_captions)
get_captions("val", train_captions, test_captions, val_captions)

# count how many times each word occur in the training set:
word_counts = {}
for img_id in train_captions:
    captions = train_captions[img_id]
    for caption in captions:
        for word in caption:
            if word not in word_counts:
                word_counts[word] = 1
            else:
                word_counts[word] += 1

# create a vocabulary of all words that appear 5 or more times in the
# training set:
vocabulary = []
for word in word_counts:
    word_count = word_counts[word]
    if word_count >= 5:
        vocabulary.append(word)

# replace all words in train that are not in the vocabulary with an 
# <UNK> token AND prepend each caption with an <SOS> token AND append 
# each caption with an <EOS> token:
for step, img_id in enumerate(train_captions):
    if step % 1000 == 0:
        print "train: ", step
        
    captions = train_captions[img_id]
    new_captions = []
    for caption in captions:
        for word_index in range(len(caption)):
            word = caption[word_index]
            if word not in vocabulary:
                caption[word_index] = "<UNK>"
        # prepend the caption with an <SOS> token;
        caption.insert(0, "<SOS>")
        # append tge caption with an <EOS> token:
        caption.append("<EOS>")
        new_captions.append(caption)
    train_captions[img_id] = new_captions

# prepend each caption in val with an <SOS> token AND append each 
# caption with an <EOS> token:
for step, img_id in enumerate(val_captions):
    if step % 1000 == 0:
        print "val: ", step
        
    captions = val_captions[img_id]
    for caption in captions:
        # prepend the caption with an <SOS> token;
        caption.insert(0, "<SOS>")
        # append tge caption with an <EOS> token:
        caption.append("<EOS>")

# prepend each caption in test with an <SOS> token AND append each 
# caption with an <EOS> token:
for step, img_id in enumerate(test_captions):
    if step % 1000 == 0:
        print "val: ", step
        
    captions = test_captions[img_id]
    for caption in captions:
        # prepend the caption with an <SOS> token;
        caption.insert(0, "<SOS>")
        # append tge caption with an <EOS> token:
        caption.append("<EOS>")
    
# save the captions to disk:    
pickle.dump(train_captions, 
        open(os.path.join(captions_dir, "train_captions"), "wb"))
pickle.dump(test_captions, 
        open(os.path.join(captions_dir, "test_captions"), "wb"))
pickle.dump(val_captions, 
        open(os.path.join(captions_dir, "val_captions"), "wb"))