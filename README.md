# CS224n_project

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
$ python classify_image.py --model_dir ~/CS224n/Project/CS224n_project/inception 

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

