# CS224n_project

Demo: http://www.fregu856.com/image_captioning  
Poster: https://goo.gl/1DMQVE  
Report: https://goo.gl/PzgRf5  

********

Installed packages (all just pip install on linux):  
numpy  
tensorflow  
Cython (for the COCO PythonAPI)  
matplotlib  
scikit-image  

********  

Clone the Tensorflow models repo: https://github.com/tensorflow/models  

Download the Inception-V3 model to where you want it (in my case to ~/CS224n/Project/CS224n_project/inception):  
$ cd models/tutorials/image/imagenet  
$ python classify_image.py --model_dir ~/CS224n/project/CS224n_project/inception 

How to extract features from the second-to-last layer of the pretrained CNN:  
https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11  

******   

Dataset: Microsoft COCO:  
http://mscoco.org/dataset/#download  

Clone/download and place the "coco" folder in your project directory:  
https://github.com/pdollar/coco  

Download the training images and place in coco/images/train:  
$ wget "http://msvocds.blob.core.windows.net/coco2014/train2014.zip"  
$ unzip train2014.zip  

Download the validation images:  
$ wget "http://msvocds.blob.core.windows.net/coco2014/val2014.zip"  
$ unzip val2014.zip  
Place 5000 of the validation images in coco/images/val, 5000 in coco/images/test and the rest in coco/images/train.  

Download the captions (captions_train2014.json and captions_val2014.json) and place in:  
coco/annotations  

To install the Python API:  
$ cd coco/PythonAPI  
$ make  

Demo of the PythonAPI:  
https://github.com/pdollar/coco/blob/master/PythonAPI/pycocoDemo.ipynb

*******

For evaluation of captions:  

Clone coco-caption and place in the coco folder in the project directory:  
https://github.com/tylin/coco-caption  
Make sure java is installed:  
$ sudo apt-get install default-jdk  

*******

For initialization of the embedding matrix with GloVe vectors:  

Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ and place glove.6B.300d.txt in coco/annotations.

# Documentation

GRU_attention_model.py:  
- ASSUMES: that preprocess_captions.py, extract_img_features_attention.py and
  create_initial_embeddings.py has already been run.
- DOES: defines the GRU_attention model and contains a script for training the
  model (basically identical to LSTM_attention_model.py).
  
********

GRU_model.py:  
- ASSUMES: that preprocess_captions.py, extract_img_features.py and
  create_initial_embeddings.py has already been run.
- DOES: defines the GRU model and contains a script for training the model (basically identical to LSTM_model.py).

*****

LSTM_attention_model.py:  
- ASSUMES: that preprocess_captions.py, extract_img_features_attention.py and
  create_initial_embeddings.py has already been run.
- DOES: defines the LSTM_attention model and contains a script for training the
  model.
  
*****

LSTM_model.py:  
- ASSUMES: that preprocess_captions.py, extract_img_features.py and
  create_initial_embeddings.py has already been run.
- DOES: defines the LSTM model and contains a script for training the model.

*********

caption_img.py:  
- Must be called in one of the following ways:
 $ caption_img.py LSTM (for using the best LSTM model)
 $ caption_img.py LSTM_attention (for using the best LSTM_attention model)
 $ caption_img.py GRU (for using the best GRU model)
 $ caption_img.py GRU_attention (for using the best GRU_attention model)
- ASSUMES: that preprocess_captions.py has already been run. That the image one
  would like to generate a caption for is called "img.jpg" and is placed in the
  directory "img_to_caption". That the weights for the best
  LSTM/GRU/LSTM_attention/GRU_attention model has been placed in
  models/**model_type**/best_model with names model.filetype.
- DOES: generates a caption for "img.jpg" using the best model of the specified
  model type and displays the img and its caption. For attention models, it also
  displays a figure visualizing the img attention at the time of prediciton for
  each word in the caption.
  
*****

caption_random_test_img.py:  
- Must be called in one of the following ways:
  $ caption_img.py LSTM [img_id] (for using the best LSTM model)
  $ caption_img.py LSTM_attention [img_id] (for using the best LSTM_attention model)
  $ caption_img.py GRU [img_id] (for using te best GRU model)
  $ caption_img.py GRU_attention [img_id] (for using the best GRU_attention model)
- ASSUMES: that preprocess_captions.py and extract_img_features.py has already
  been run. That the weights for the best LSTM/GRU/LSTM_attention/GRU_attention
  model has been placed in models/**model_type**/best_model with names
  model.filetype.
- DOES: generates a caption for the test img with img id img_id if specified,
  otherwise for a random test img. It also displays the img and its caption.
  For attention models, it also displays a figure visualizing the img attention
  at the time of prediciton for each word in the caption.
  
*****

create_initial_embeddings.py:  
- ASSUMES: that "preprocess_captions.py" already has been run.
- DOES: creates a word embedding matrix (embeddings_matrix) using GloVe vectors.

******

evaluate_best_models_on_test.py:  
- ASSUMES: that preprocess_captions.py, extract_img_features.py and
  extract_img_features_attention.py has already been run. That the weights for the
  best LSTM/GRU/LSTM_attention/GRU_attention model has been placed in
  models/**model_type**/best_model with names model.filetype.
- DOES: generates captions for all 5000 imgs in test using the best
  LSTM/GRU/LSTM_attention/GRU_attention model, evaluates the captions and
  returns the metric scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, CIDEr, METEOR and
  ROUGE_L).
  
****

extract_img_features.py:  
- ASSUMES: that the image dataset has been manually split such that all train
  images are stored in "coco/images/train/", all test images are stored in
  "coco/images/test/" and all val images are stored in "coco/images/val". That
  the Inception-V3 model has been downloaded and placed in inception.
- DOES: extracts a 2048 dimensional feature vector for each train/val/test img
  and creates dicts mapping from img id to feature vector (
  train/val/test_img_id_2_feature_vector).

****

extract_img_features_attention.py:  
- ASSUMES: that the image dataset has been manually split such that all train
  images are stored in "coco/images/train/", all test images are stored in
  "coco/images/test/" and all val images are stored in "coco/images/val". That
  the Inception-V3 model has been downloaded and placed in inception. That the
  dict numpy_params (containing W_img and b_img taken from the img_transform
  step in a well-performing non-attention model) is placed in
  coco/data/img_features_attention/transform_params.
- DOES: extracts a 64x300 feature array (64 300 dimensional feature vectors,
  one each for 8x8 different img regions) for each train/val/test img and saves
  each individual feature array to disk (to coco/data/img_features_attention).
  Is used in the attention models.
  
******

preprocess_captions.py:  
- ASSUMES: that "split_img_ids.py" already has been run. That the COCO Python API
  has been installed. That the files captions_train2014.json,
  captions_val2014.json and glove.6B.300d.txt is placed in coco/annotations.
  That the folder coco/data exists.
- DOES: all necessary pre-processing of the captions. Creates a number of files,
  see all "cPickle.dump" below. 
  
****

split_img_ids.py:  
- ASSUMES: that the image dataset has been manually split such that all test
  images are stored in "coco/images/test/" and all val images are stored in
  "coco/images/val".
- DOES: creates two files (val_img_ids, test_img_ids) containing the img ids for
  all val and test imgs, respectively. Is later used to sort an img as either
  train, val or test.
  
*****

test.py:  
- DOES: contains a bunch of code snippets that have been tested or used at some
  point. Probably nothing interesting to see here.

*****

utilities.py:  
- DOES: contains a number of functions used in different parts of the project.

****

web/app.py:  
- DOES: contains backend code for local live demo webpage.

****

web/templates/index.html:  
- DOES: contains frontend code for local live demo webpage.
