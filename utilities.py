import cPickle
import numpy as np
import os

def get_batches(batch_size):
    caption_length_2_caption_ids =\
            cPickle.load(open("coco/data/train_caption_length_2_caption_ids"))
    caption_length_2_no_of_captions =\
            cPickle.load(open("coco/data/train_caption_length_2_no_of_captions"))

    # group all caption ids in batches:
    batches_of_captions = []
    for caption_length in caption_length_2_no_of_captions:
        caption_ids = caption_length_2_caption_ids[caption_length]
        no_of_captions = caption_length_2_no_of_captions[caption_length]
        no_of_full_batches = int(no_of_captions/batch_size)

        # add all full batches to batches_of_captions:
        for i in range(no_of_full_batches):
            batch_caption_ids = caption_ids[i*batch_size:(i+1)*batch_size]
            batches_of_captions.append(batch_caption_ids)

        # get the remaining caption ids and add to batches_of_captions (not a
        # full batch, i.e, it will contain fewer than "batch_size" captions):
        #batch_caption_ids = caption_ids[no_of_full_batches*batch_size:]
        #batches_of_captions.append(batch_caption_ids)

    return batches_of_captions

batches = get_batches(256)

print batches
