import pickle
import os

captions_dir = "coco/annotations/"
ids_dir = "coco/features/"

# load the captions from disk:        
captions = pickle.load(open(os.path.join(captions_dir, "captions")))

# load the test image ids from disk:        
test_img_ids = pickle.load(open(os.path.join(ids_dir, "img_ids_test")))

img_id = int(test_img_ids[0])
img_captions = captions[img_id]

print img_captions

for caption in img_captions:
    print caption