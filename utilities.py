import pickle
import os
import numpy as np

def tokenize_caption(caption):
    # load the vocabulary from disk:  
    captions_dir = "coco/annotations/"    
    vocabulary = pickle.load(open(os.path.join(captions_dir, 
                                               "vocabulary"))) 
    
    # tokenize the caption:
    tokenized_caption = []
    for word in caption:
        if word == "<START>":
            tokenized_caption.append(-1)
        elif word == "<UNK>":
            tokenized_caption.append(-2)
        else:
            word_index = vocabulary.index(word)
            tokenized_caption.append(word_index)
    
    # convert into a numpy array:
    tokenized_caption = np.array(tokenized_caption)
    
    return tokenized_caption