import cPickle
import os
import numpy as np

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

#test_img_ids = cPickle.load(open("coco/data/test_img_ids"))

#val_img_ids = cPickle.load(open("coco/data/val_img_ids"))

#caption_id_2_img_id = cPickle.load(open("coco/data/caption_id_2_img_id"))

#test_caption_id_2_caption = cPickle.load(open("coco/data/test_caption_id_2_caption"))

#train_caption_id_2_caption = cPickle.load(open("coco/data/train_caption_id_2_caption"))

#val_caption_id_2_caption = cPickle.load(open("coco/data/val_caption_id_2_caption"))

#vocabulary = cPickle.load(open("coco/data/vocabulary"))

#embeddings = cPickle.load(open("coco/data/embeddings_matrix"))

#val_img_id_2_feature_vector = cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
#test_img_id_2_feature_vector = cPickle.load(open("coco/data/test_img_id_2_feature_vector"))
#train_img_id_2_feature_vector = cPickle.load(open("coco/data/train_img_id_2_feature_vector"))

#caption_id = 829719
#img_id = caption_id_2_img_id[caption_id]

#print test_img_id_2_feature_vector[img_id]

#batches_of_captions = cPickle.load(open("coco/data/batches_of_captions"))
#print batches_of_captions
#print caption_id_2_img_id[batches_of_captions[127][123]]

# # tokenize all test captions:
# for step, caption_id in enumerate(test_caption_id_2_caption):
#     if step % 1000 == 0:
#         print "test, tokenizing: ", step
#
#     caption = test_caption_id_2_caption[caption_id]
#
#     # tokenize the caption:
#     tokenized_caption = []
#     for word in caption:
#         if word in vocabulary:
#             word_index = vocabulary.index(word)
#         else:
#             word_index = -99
#         tokenized_caption.append(word_index)
#
#     # convert into a numpy array:
#     tokenized_caption = np.array(tokenized_caption)
#     # save:
#     test_caption_id_2_caption[caption_id] = tokenized_caption
#
# # save all the captions to disk:
# cPickle.dump(test_caption_id_2_caption, open(os.path.join("coco/data/",
#         "testing"), "wb"))
