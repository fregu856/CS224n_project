"""
- ASSUMES: that "preprocess_captions.py" already has been run.

- DOES: creates a word embedding matrix (embeddings_matrix) using GloVe vectors.
"""

import numpy as np
import cPickle
import os
from utilities import log

captions_dir = "coco/annotations/"
data_dir = "coco/data/"
word_vec_dim = 300

# load the vocabulary from disk:
vocabulary = cPickle.load(open(os.path.join(data_dir, "vocabulary")))
vocab_size = len(vocabulary)

# read all words and their corresponding pretrained word vec from file:
pretrained_words = []
word_vectors = []
with open(os.path.join(captions_dir, "glove.6B.300d.txt")) as file:
    for line in file:
        # remove the new line char at the end:
        line = line.strip()

        # seperate the word from the word vector:
        line_elements = line.split(" ")
        word = line_elements[0]
        word_vector = line_elements[1:]

        # save:
        pretrained_words.append(word)
        word_vectors.append(word_vector)

# create an embedding matrix where row i is the pretrained word vector
# corresponding to word i in the vocabulary:
embeddings_matrix = np.zeros((vocab_size, word_vec_dim))
for vocab_index, word in enumerate(vocabulary):
    if vocab_index % 1000 == 0:
        print vocab_index
        log(str(vocab_index))

    if word not in ["<SOS>", "<UNK>", "<EOS>"]: # (the special tokens are initialized with zero vectors)
        word_embedd_index = pretrained_words.index(word)
        word_vector = word_vectors[word_embedd_index]
        # convert into a numpy array:
        word_vector = np.array(word_vector)
        # convert everything to floats:
        word_vector = word_vector.astype(float)
        # add to the matrix:
        embeddings_matrix[vocab_index, :] = word_vector

# save the embeddings_matrix to disk:
cPickle.dump(embeddings_matrix,
        open(os.path.join(data_dir, "embeddings_matrix"), "wb"))
