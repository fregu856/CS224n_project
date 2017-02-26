"""
- ASSUMES: that preprocess_captions.py, extract_img_features.py and
  create_initial_embeddings.py has already been run.

- DOES: defines the LSTM model and contains a script for training the model.
"""

import numpy as np
import tensorflow as tf

import cPickle
import os
import time
import json
import cPickle
import random

from utilities import train_data_iterator, detokenize_caption, evaluate_captions
from utilities import plot_performance, log

class LSTM_Config(object):
    """
    - DOES: config object containing a number of parameters.
    """

    def __init__(self, debug=False):
        self.dropout = 0.5 # (keep probability)
        self.embed_dim = 300 # (dimension of word embeddings)
        self.hidden_dim = 200 # (dimension of hidden state)
        self.batch_size = 256
        self.lr = 0.001
        self.img_dim = 2048 # (dimension of img feature vectors)
        self.vocab_size = 9855 # (no of words in the vocabulary)
        self.no_of_layers = 3 # (no of layers in the RNN)
        if debug:
            self.max_no_of_epochs = 2
        else:
            self.max_no_of_epochs = 100
        self.max_caption_length = 40
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim, self.embed_dim,
                    self.no_of_layers)
        self.model_dir = "models/LSTMs/%s" % self.model_name

class LSTM_Model(object):
    """
    - DOES: defines the LSTM model.
    """

    def __init__(self, config, GloVe_embeddings, debug=False, mode="training"):
        """
        - DOES: initializes some parameters and adds everything to the
        computational graph.
        """

        self.GloVe_embeddings = GloVe_embeddings
        self.debug = debug
        self.config = config
        if mode is not "demo":
            # create all dirs for saving weights and eval results:
            self.create_model_dirs()
            # load all data from disk needed for training:
            self.load_utilities_data()
        # add placeholders to the comp graph:
        self.add_placeholders()
        # transform the placeholders and add the final model input to the graph:
        self.add_input()
        # compute logits (unnormalized prediction probs) and add to the graph:
        self.add_logits()
        if mode is not "demo":
            # compute the loss and add to the graph:
            self.add_loss_op()
            # add a training operation for optimizing the loss to the graph:
            self.add_training_op()

    def create_model_dirs(self):
        """
        - DOES: creates all model directories needed for saving weights, losses,
        evaluation metrics etc. during training.
        """

        # create the main model directory:
        if not os.path.exists(self.config.model_dir):
            os.mkdir(self.config.model_dir)

        # create the dir where model weights will be saved during training:
        if not os.path.exists("%s/weights" % self.config.model_dir):
            os.mkdir("%s/weights" % self.config.model_dir)

        # create the dir where generated captions will be saved during training:
        if not os.path.exists("%s/generated_captions" % self.config.model_dir):
            os.mkdir("%s/generated_captions" % self.config.model_dir)

        # create the dir where epoch losses will be saved during training:
        if not os.path.exists("%s/losses" % self.config.model_dir):
            os.mkdir("%s/losses" % self.config.model_dir)

        # create the dir where evaluation metrics will be saved during training:
        if not os.path.exists("%s/eval_results" % self.config.model_dir):
            os.mkdir("%s/eval_results" % self.config.model_dir)

        # create the dir where performance plots will be saved after training:
        if not os.path.exists("%s/plots" % self.config.model_dir):
            os.mkdir("%s/plots" % self.config.model_dir)

    def load_utilities_data(self):
        """
        - DOES: loads all data from disk (vocabulary, img feature vectors etc.)
        needed for training.
        """

        print "loading utilities data..."
        log("loading utilities data...")

        # load the vocabulary:
        self.vocabulary = cPickle.load(open("coco/data/vocabulary"))

        # load data to map from caption id to img feature vector:
        self.caption_id_2_img_id =\
                    cPickle.load(open("coco/data/caption_id_2_img_id"))
        if self.debug:
            self.train_img_id_2_feature_vector =\
                    cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
        else:
            self.train_img_id_2_feature_vector =\
                    cPickle.load(open("coco/data/train_img_id_2_feature_vector"))

        # load data to map from caption id to caption:
        if self.debug:
            self.train_caption_id_2_caption =\
                    cPickle.load(open("coco/data/val_caption_id_2_caption"))
        else:
            self.train_caption_id_2_caption =\
                    cPickle.load(open("coco/data/train_caption_id_2_caption"))

        # load data needed to create batches:
        if self.debug:
            self.caption_length_2_caption_ids =\
                cPickle.load(open("coco/data/val_caption_length_2_caption_ids"))
            self.caption_length_2_no_of_captions =\
                cPickle.load(open("coco/data/val_caption_length_2_no_of_captions"))
        else:
            self.caption_length_2_caption_ids =\
                cPickle.load(open("coco/data/train_caption_length_2_caption_ids"))
            self.caption_length_2_no_of_captions =\
                cPickle.load(open("coco/data/train_caption_length_2_no_of_captions"))

        print "all utilities data is loaded!"
        log("all utilities data is loaded!")

    def add_placeholders(self):
        """
        - DOES: adds placeholders for captions, imgs, labels and keep_prob to
        the computational graph. These placeholders will be fed actual data
        corresponding to each batch during training.
        """

        # add the placeholder for the batch captions (row i of caption_ph will
        # be the tokenized caption for ex i in the batch):
        self.captions_ph = tf.placeholder(tf.int32,
                    shape=[None, None], # ([batch_size, caption_length])
                    name="captions_ph")
        # add the placeholder for the batch imgs (row i of imgs_ph will be the
        # img feature vector for ex i in the batch):
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[None, self.config.img_dim], # ([batch_size, img_dim])
                    name="imgs_ph")
        # add the placeholder for the batch labels (row i of labels_ph will
        # be the labels/targets for ex i in the batch):
        self.labels_ph = tf.placeholder(tf.int32,
                    shape=[None, None], # ([batch_size, caption_length+1])
                    name="labels_ph")
        # add the placeholder for the keep_prob (with what probability we will
        # keep a hidden unit during training):
        self.dropout_ph = tf.placeholder(tf.float32, name="dropout_ph") # (keep_prob)

    def create_feed_dict(self, captions_batch, imgs_batch, labels_batch=None, dropout=1):
        """
        - DOES: returns a feed_dict mapping the placeholders to the actual
        input data (this is how we run the network on specific data).
        """

        feed_dict = {}
        feed_dict[self.captions_ph] = captions_batch
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.dropout_ph] = dropout
        if labels_batch is not None:
            # only add the labels data if it's specified (during caption
            # generation, we won't have any labels):
            feed_dict[self.labels_ph] = labels_batch

        return feed_dict

    def add_input(self):
        """
        - DOES: transforms the imgs_ph to a tensor of shape
        [batch_size, 1, embed_dim], gets the word vector for each tokenized word
        in captions_ph giving a tensor of shape
        [batch_size, caption_length, embed_dim], and finally concatenates the
        two into a tensor of shape [batch_size, caption_length+1, embed_dim].
        This tensor is the input to the network, meaning that we will feed in
        the img, then <SOS>, then each word in the caption, and then
        finally <EOS>.
        """

        # transform img_ph into a tensor of shape [batch_size, 1, embed_dim]:
        with tf.variable_scope("img_transform"):
            # initialize the transform parameters:
            W_img = tf.get_variable("W_img",
                        shape=[self.config.img_dim, self.config.embed_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_img = tf.get_variable("b_img", shape=[1, self.config.embed_dim],
                        initializer=tf.constant_initializer(0))
            # tranform img_ph to shape [batch_size, embed_dim]:
            imgs_input = tf.nn.sigmoid(tf.matmul(self.imgs_ph, W_img) + b_img)
            # reshape into shape [batch_size, 1, embed_dim]:
            imgs_input = tf.expand_dims(imgs_input, 1)

        # get the word vector for each tokenized word in captions_ph:
        with tf.variable_scope("captions_embed"):
            # initialize the embeddings matrix with pretrained GloVe vectors (
            # note that we will train the embeddings matrix as well!):
            word_embeddings = tf.get_variable("word_embeddings",
                        initializer=self.GloVe_embeddings)
            # get the word vectors (gives a tensor of shape
            # [batch_size, caption_length, embed_dim]):
            captions_input = tf.nn.embedding_lookup(word_embeddings,
                        self.captions_ph)

        # concatenate imgs_input and captions_input to get the final input (has
        # shape [batch_size, caption_length+1, embed_dim])
        self.input = tf.concat(1, [imgs_input, captions_input])

    def add_logits(self):
        """
        - DOES: feeds self.input through an LSTM, producing a hidden state vector
        for each word/img and computes all corresponding logits (unnormalized
        prediction probabilties over the vocabulary, a softmax step but without
        the actual softmax).
        """

        # create an LSTM cell:
        LSTM_cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim)
        # apply dropout to the LSTM cell:
        LSTM_cell = tf.nn.rnn_cell.DropoutWrapper(LSTM_cell,
                    input_keep_prob=self.dropout_ph,
                    output_keep_prob=self.dropout_ph)
        # stack no_of_layers LSTM cells on top of each other (for a deep LSTM):
        stacked_LSTM_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [LSTM_cell]*self.config.no_of_layers)
        # initialize the state of the stacked LSTM cell (tf.shape(self.input)[0]
        # gets the current batch size) (the state contains both h and c for all
        # layers, thus its format is not trivial):
        initial_state = stacked_LSTM_cell.zero_state(tf.shape(self.input)[0],
                    tf.float32)

        # feed self.input trough the stacked LSTM cell and get the (top) hidden
        # state vector for each word/img returned in outputs (which has shape
        # [batch_size, caption_length+1, hidden_dim]) (final_state contains
        # h and c for all layers at the final timestep, not relevant here):
        outputs, final_state = tf.nn.dynamic_rnn(stacked_LSTM_cell,
                    self.input, initial_state=initial_state)
        # reshape outputs into shape [batch_size*(caption_length+1), hidden_dim]
        # (outputs[0]: h for the img in ex 1 in the batch, outputs[1]: h for <SOS>
        # in ex 1 in the batch etc.):
        outputs = tf.reshape(outputs, [-1, self.config.hidden_dim])

        # compute corresponding logits for each hidden state vector in outputs,
        # resulting in a tensor self.logits of shape
        # [batch_size*(caption_length+1), vocab_size] (each word in self.input
        # will have a corr. logits vector, which is an unnorm. prob. distr. over
        # the vocab. The largets element corresponds to the predicted next word):
        with tf.variable_scope("logits"):
            # initialize the transform parametrs:
            W_logits = tf.get_variable("W_logits",
                        shape=[self.config.hidden_dim, self.config.vocab_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_logits = tf.get_variable("b_logits",
                        shape=[1, self.config.vocab_size],
                        initializer=tf.constant_initializer(0))
            # compute the logits:
            self.logits = tf.matmul(outputs, W_logits) + b_logits

    def add_loss_op(self):
        """
        - DOES: computes the CE loss for the batch.
        """

        # reshape labels_ph into shape [batch_size*(caption_length+1), ] (to
        # match the shape of self.logits):
        labels = tf.reshape(self.labels_ph, [-1])
        print labels.get_shape()

        # remove all -1 labels and their corresponding logits (-1 labels
        # correspond to the img or <EOS> step, the predicitons at these
        # steps are irrelevant and should not contribute to the loss):
        mask = tf.greater_equal(labels, 0)
        masked_labels = tf.boolean_mask(labels, mask)
        masked_logits = tf.boolean_mask(self.logits, mask)

        loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    masked_logits, masked_labels)
        loss = tf.reduce_mean(loss_per_word)

        self.loss = loss

    def add_training_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def run_epoch(self, session):
        batch_losses = []

        for step, (captions, imgs, labels) in enumerate(train_data_iterator(self)):
            feed_dict = self.create_feed_dict(captions, imgs,
                        labels_batch=labels, dropout=self.config.dropout)
            batch_loss, _ = session.run([self.loss, self.train_op],
                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)

            if step % 100 == 0:
                print "batch: %d | loss: %f" % (step, batch_loss)
                log("batch: %d | loss: %f" % (step, batch_loss))

            if step > 5 and self.debug:
                break

        return batch_losses

    def generate_img_caption(self, session, img_vector, vocabulary):
        # initialize the caption as "<SOS>":
        caption = np.zeros((1, 1))
        caption[0] = np.array(vocabulary.index("<SOS>"))
        # format the img_vector so it can be fed to the NN:
        img = np.zeros((1, self.config.img_dim))
        img[0] = img_vector
        # we will get one vector of logits for each timestep, 0: img, 1: "<SOS>",
        # we want to get the one corr. to "<SOS>":
        prediction_index = 1

        # predict the next word given the img and the current caption until we
        # get "<EOS>" or the caption length hits a max value:
        while int(caption[0][-1]) is not vocabulary.index("<EOS>") and\
                    caption.shape[1] < self.config.max_caption_length:
            feed_dict = self.create_feed_dict(caption, img)
            logits = session.run(self.logits, feed_dict=feed_dict)
            # (logits[0] = logits vector corr. to the img in ex #1 in the batch,
            # logits[1] = logits vector corr. to <SOS> in ex #1 in the batch, etc)
            # get the logits vector corr. to the last word in the current caption:
            prediction_logits = logits[prediction_index]
            # get the index of the predicted word:
            predicted_word_index = np.argmax(prediction_logits)
            # add the new word to the caption (only care about the first row):
            new_word_col = np.zeros((1, 1))
            new_word_col[0] = np.array(predicted_word_index)
            caption = np.append(caption, new_word_col, axis=1)
            # increment prediction_index so that we'll look at the new last word
            # of the caption in the next iteration:
            prediction_index += 1

        # get the caption and convert to ints:
        caption = caption[0].astype(int)
        # convert the caption to actual text:
        caption = detokenize_caption(caption, vocabulary)

        return caption

    def generate_captions_on_val(self, session, epoch, vocabulary, val_set_size=5000):
        if self.debug:
            val_set_size = 101

        # get the map from img id to feature vector:
        val_img_id_2_feature_vector =\
                    cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
        # turn the map into a list of tuples (to make it iterable):
        val_img_id_feature_vector_list = val_img_id_2_feature_vector.items()
        # randomly shuffle the list of tuples (to take different subsets when
        # val_set_size is not set to 5000):
        #random.shuffle(val_img_id_feature_vector_list)
        # take a subset (of size val_set_size) of all val imgs:
        val_set = val_img_id_feature_vector_list[0:val_set_size]

        captions = []
        for step, (img_id, img_vector) in enumerate(val_set):
            if step % 100 == 0:
                print "generating captions on val: %d" % step
                log("generating captions on val: %d" % step)

            # generate a caption for the img:
            img_caption = self.generate_img_caption(session, img_vector, vocabulary)
            # save the generated caption together with the img id in the format
            # expected by the COCO evaluation script:
            caption_obj = {}
            caption_obj["image_id"] = img_id
            caption_obj["caption"] = img_caption
            captions.append(caption_obj)

        # save the captions as a json file (will be used by the eval script):
        captions_file = "%s/generated_captions/captions_%d.json"\
                    % (self.config.model_dir, epoch)
        with open(captions_file, "w") as file:
            json.dump(captions, file, sort_keys=True, indent=4)

        # return the name of the json file:
        return captions_file

def main():
    config = LSTM_Config()
    GloVe_embeddings = cPickle.load(open("coco/data/embeddings_matrix"))
    GloVe_embeddings = GloVe_embeddings.astype(np.float32)
    model = LSTM_Model(config, GloVe_embeddings)

    loss_per_epoch = []
    eval_metrics_per_epoch = []

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=model.config.max_no_of_epochs)

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(config.max_no_of_epochs):
            print "###########################"
            print "######## NEW EPOCH ########"
            print "###########################"
            print "epoch: %d/%d" % (epoch, config.max_no_of_epochs-1)
            log("###########################")
            log("######## NEW EPOCH ########")
            log("###########################")
            log("epoch: %d/%d" % (epoch, config.max_no_of_epochs-1))

            # run an epoch and get all losses:
            batch_losses = model.run_epoch(sess)

            # compute the epoch loss:
            epoch_loss = np.mean(batch_losses)
            # save the epoch loss:
            loss_per_epoch.append(epoch_loss)
            # save the epoch losses to disk:
            cPickle.dump(loss_per_epoch, open("%s/losses/loss_per_epoch"\
                        % model.config.model_dir, "w"))

            # generate captions on a (subset) of val:
            captions_file = model.generate_captions_on_val(sess, epoch,
                        model.vocabulary, val_set_size=1000)
            # evaluate the generated captions (compute metrics):
            eval_result_dict = evaluate_captions(captions_file)
            # save the epoch evaluation metrics:
            eval_metrics_per_epoch.append(eval_result_dict)
            # save the evaluation metrics for epochs to disk:
            cPickle.dump(eval_metrics_per_epoch, open("%s/eval_results/metrics_per_epoch"\
                        % model.config.model_dir, "w"))

            # save the model weights to disk:
            saver.save(sess, "%s/weights/model" % model.config.model_dir,
                        global_step=epoch)

            print "epoch loss: %f | BLEU4: %f  |  CIDEr: %f" % (epoch_loss, eval_result_dict["Bleu_4"], eval_result_dict["CIDEr"])
            log("epoch loss: %f | BLEU4: %f  |  CIDEr: %f" % (epoch_loss, eval_result_dict["Bleu_4"], eval_result_dict["CIDEr"]))

    # plot the loss and the different metrics vs epoch:
    plot_performance(config.model_dir)

if __name__ == '__main__':
    main()
