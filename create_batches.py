## PLEASE NOTE!
# this file is not used, this is done in preprocess_captions.py instead

"""
- Assumes that "preprocess_captions.py" already has been run.
"""

import cPickle
import os

# get the train captions:
train_caption_id_2_caption = cPickle.load(open("coco/data/train_caption_id_2_caption"))

# map all train captions to their length:
train_caption_length_2_caption_ids = {}
for caption_id in train_caption_id_2_caption:
    caption = train_caption_id_2_caption[caption_id]
    caption_length = len(caption)
    if caption_length not in train_caption_length_2_caption_ids:
        train_caption_length_2_caption_ids[caption_length] = [caption_id]
    else:
        train_caption_length_2_caption_ids[caption_length].append(caption_id)

# map each train caption length to the number of captions of that length:
train_caption_length_2_no_of_captions = {}
for caption_length in train_caption_length_2_caption_ids:
    caption_ids = train_caption_length_2_caption_ids[caption_length]
    no_of_captions = len(caption_ids)
    train_caption_length_2_no_of_captions[caption_length] = no_of_captions

# save to disk (this is needed when creating batches for training):
cPickle.dump(train_caption_length_2_no_of_captions,
        open(os.path.join("coco/data/",
        "train_caption_length_2_no_of_captions"), "wb"))
cPickle.dump(train_caption_length_2_caption_ids,
        open(os.path.join("coco/data/",
        "train_caption_length_2_caption_ids"), "wb"))
