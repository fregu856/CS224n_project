import numpy as np
import pickle
import os
import re

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/afs/.ir.stanford.edu/users/f/r/fregu856/CS224n/Project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

captions_dir = "coco/annotations/"

def get_captions(type_of_data, captions):
    captions_file = "coco/annotations/captions_%s2014.json" % type_of_data

    # initialize COCO api for captions:
    coco=COCO(captions_file)

    # get indices for all "type_of_data" images (all train or val images):
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
            
        # store the captions in the dict "captions":
        captions[img_id] = captions_vec

captions = {}
get_captions("train", captions)
get_captions("val", captions)
    
# save the captions to disk:    
pickle.dump(captions, 
        open(os.path.join(captions_dir, "captions"), "wb"))

# load the captions from disk:        
#test = pickle.load(open(os.path.join(captions_dir, "captions")))
#print test

