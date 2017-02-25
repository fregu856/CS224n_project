"""
- ASSUMES: that the image dataset has been manually split such that all test
  images are stored in "coco/images/test/" and all val images are stored in
  "coco/images/val".

- DOES: creates two files (val_img_ids, test_img_ids) containing the img ids for
  all val and test imgs, respectively. Is later used to sort an img as either
  train, val or test.
"""

import cPickle
import os
import numpy as np

# define where all val images are located:
val_img_dir = "coco/images/val/"
# create a list of the paths to all val images:
val_img_paths = [val_img_dir + file_name for file_name in\
                 os.listdir(val_img_dir) if ".jpg" in file_name]
# define where all test images are located:
test_img_dir = "coco/images/test/"
# create a list of the paths to all test images:
test_img_paths = [test_img_dir + file_name for file_name in\
                  os.listdir(test_img_dir) if ".jpg" in file_name]

# get all val img ids:
val_img_ids = np.array([])
for val_img_path in val_img_paths:
    img_name = val_img_path.split("/")[3]
    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
    img_id = int(img_id)
    val_img_ids = np.append(val_img_ids, img_id)

# get all test img ids:
test_img_ids = np.array([])
for test_img_path in test_img_paths:
    img_name = test_img_path.split("/")[3]
    img_id = img_name.split("_")[2].split(".")[0].lstrip("0")
    img_id = int(img_id)
    test_img_ids = np.append(test_img_ids, img_id)

# save the val img ids to disk:
cPickle.dump(val_img_ids, open(os.path.join("coco/data/", "val_img_ids"), "wb"))
# save the test img ids to disk:
cPickle.dump(test_img_ids, open(os.path.join("coco/data/", "test_img_ids"), "wb"))
