import cPickle
import os

#captions_dir = "coco/annotations/"
#ids_dir = "coco/features/"

# load the captions from disk:        
#test_captions = pickle.load(open(os.path.join(captions_dir, "test_captions")))

# load the test image ids from disk:        
#test_img_ids = pickle.load(open(os.path.join(ids_dir, "img_ids_test")))

#img_id = int(test_img_ids[0])
#img_captions = test_captions[img_id]

#print img_captions

#for caption in img_captions:
#    print caption
    
# load the vocabulary from disk:        
#vocabulary = pickle.load(open(os.path.join(captions_dir, "vocabulary")))

#print vocabulary

# load the embeddings matrix from disk:        
#embeddings_matrix = pickle.load(open(os.path.join(captions_dir, "embeddings_matrix")))

#print embeddings_matrix

test_img_ids = cPickle.load(open("coco/data/test_img_ids"))

val_img_ids = cPickle.load(open("coco/data/val_img_ids"))

print test_img_ids
print len(val_img_ids)