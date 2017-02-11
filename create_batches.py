import cPickle
import numpy as np

batch_size = 256

# get the train captions:
train_caption_id_2_caption = cPickle.load(open("coco/data/train_caption_id_2_caption"))

# map all captions to their length:
caption_length_2_caption_ids = {}
for caption_id in train_caption_id_2_caption:
    caption = train_caption_id_2_caption[caption_id]
    caption_length = len(caption)
    if caption_length not in caption_length_2_caption_ids:
        caption_length_2_caption_ids[caption_length] = [caption_id]
    else:
        caption_length_2_caption_ids[caption_length].append(caption_id)

# map each caption length to the number of captions of that length:
caption_length_2_no_of_captions = {}
for caption_length in caption_length_2_caption_ids:
    caption_ids = caption_length_2_caption_ids[caption_length]
    no_of_captions = len(caption_ids)
    caption_length_2_no_of_captions[caption_length] = no_of_captions

# group all caption ids in batches:
batches = []
for caption_length in caption_length_2_no_of_captions:
    caption_ids = caption_length_2_caption_ids[caption_length]
    no_of_captions = caption_length_2_no_of_captions[caption_length]
    no_of_full_batches = int(no_of_batches/batch_size)

    # add all full batches to batches:
    for i in range(no_of_full_batches):
        batch_caption_ids = caption_ids[i*batch_size:(i+1)*batch_size]
        batches.append(batch_caption_ids)

    # get the remaining caption ids and add to batches (not a full batch, i.e,
    # it will contain fewer than "batch_size" captions):
    batch_caption_ids = caption_ids[no_of_full_batches*batch_size:]
    batches.append(batch_caption_ids)
