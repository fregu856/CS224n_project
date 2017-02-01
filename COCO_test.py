import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/afs/.ir.stanford.edu/users/f/r/fregu856/CS224n/Project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

type_of_data = "train" # ("train" or "val")
captions_file = "coco/annotations/captions_%s2014.json" % type_of_data

# initialize COCO api for captions:
coco=COCO(captions_file)

# get indices for all "type_of_data" images (all train or val images):
img_ids = coco.getImgIds()

for img_id in img_ids:
    # get the id for one image:
    #img_id = img_ids[0]
    
    # get the corresponding image object:
    img = coco.loadImgs(img_id)[0]
    # download the corresponding image:
    coco.download(tarDir = "coco/images", imgIds=[img_id])

# get the ids of all captions for the image:
#caption_ids = coco.getAnnIds(imgIds=img["id"])
# get all caption objects for the image:
#captions = coco.loadAnns(caption_ids)
# print all captions:
#coco.showAnns(captions)

#print "*******"

# print all captions in a different way:
#for caption_obj in captions:
#    caption = caption_obj["caption"]
#    print caption
    
#print len(img_ids)