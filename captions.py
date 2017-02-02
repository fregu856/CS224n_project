import numpy as np
import pickle
import os

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
    img_ids = img_ids[0:10]

    for img_id in img_ids:
        # get the ids of all captions for the image:
        caption_ids = coco.getAnnIds(imgIds=img_id)
        # get all caption objects for the image:
        caption_objs = coco.loadAnns(caption_ids)
        
        # get all captions for the image:
        captions_vec = []
        for caption_obj in caption_objs:
            caption = caption_obj["caption"]
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
test = pickle.load(open(os.path.join(captions_dir, "captions")))
print test

