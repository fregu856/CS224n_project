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
            # make the caption lower case:
            caption = caption.lower()
            # remove all non-alphanum chars (keep spaces between word):
            caption = re.sub("[^a-z0-9 ]+", "", caption)
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

# replace all words (in train) that appears less than five times 
# (in train) with an UNK token:

# append each caption with an SOS token:

# prepend each caption with an EOS token:
    
# save the captions to disk:    
pickle.dump(train_captions, 
        open(os.path.join(captions_dir, "train_captions"), "wb"))
pickle.dump(test_captions, 
        open(os.path.join(captions_dir, "test_captions"), "wb"))
pickle.dump(val_captions, 
        open(os.path.join(captions_dir, "val_captions"), "wb"))