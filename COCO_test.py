import numpy as np
#import skimage.io as io
import matplotlib.pyplot as plt

# add the "PythonAPI" dir to the path so that "pycocotools" can be found:
import sys
sys.path.append("/afs/.ir.stanford.edu/users/f/r/fregu856/CS224n/Project/CS224n_project/coco/PythonAPI")
from pycocotools.coco import COCO

dataDir = "coco"
dataType = "train2014"
annFile = "%s/annotations/captions_%s.json" %(dataDir, dataType)

# initialize the COCO api for instance annotations:
coco = COCO(annFile)