"""
- ASSUMES: that preprocess_captions.py, extract_img_features_attention.py and
  create_initial_embeddings.py has already been run.

- DOES: defines the GRU_attention model and contains a script for training the
  model.
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
from utilities import plot_performance, compare_captions, log
from utilities import train_data_iterator_attention, get_max_caption_length

class GRU_attention_Config(object):
    """
    - DOES: config object containing a number of model parameters.
    """

    def __init__(self, debug=False):
        self.dropout = 0.75 # (keep probability)
        self.embed_dim = 300 # (dimension of word embeddings)
        self.hidden_dim = 400 # (dimension of hidden state)
        self.batch_size = 256
        self.lr = 0.001
        self.img_feature_dim = 300 # (dim of img feature vectors)
        self.no_of_img_feature_vecs = 64 # (no of feature vectors per img)
        self.vocab_size = 9855 # (no of words in the vocabulary)
        self.no_of_layers = 1 # (no of layers in the RNN)
        self.hidden_dim_att = 500 # (dim of hidden state in the attention network)
        if debug:
            self.max_no_of_epochs = 2
        else:
            self.max_no_of_epochs = 120
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d_hidden_dim_att=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim, self.embed_dim,
                    self.no_of_layers, self.hidden_dim_att)
        self.model_dir = "models/GRUs_attention/%s" % self.model_name
        self.max_caption_length = get_max_caption_length(self.batch_size)

class GRU_attention_Model(object):
    """
    - DOES: defines the GRU_attention model.
    """

    def __init__(self, config, GloVe_embeddings, debug=False, mode="training"):
        """
        - DOES: initializes some parameters and adds everything to the
        computational graph.
        """

        self.GloVe_embeddings = GloVe_embeddings
        self.debug = debug
        self.config = config
        self.mode = mode
        if mode is not "demo":
            # create all dirs for saving weights and eval results:
            self.create_model_dirs()
            # load all data from disk needed for training:
            self.load_utilities_data()
        # add placeholders to the comp graph:
        self.add_placeholders()
        # get word vectors for all words i caption_ph and add to the graph:
        self.add_captions_input()
        # compute logits (unnormalized prediction probs) and add to the graph:
        self.add_logits()
        if mode is not "demo":
            # compute the batch loss and add to the graph:
            self.add_loss_op()
            # add a training operation (for optimizing the loss) to the graph:
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
        - DOES: loads all data from disk (vocabulary etc.) needed for training.
        """

        print "loading utilities data..."
        log("loading utilities data...")

        # load the vocabulary:
        self.vocabulary = cPickle.load(open("coco/data/vocabulary"))

        # load data to map from caption id to img id:
        self.caption_id_2_img_id =\
                    cPickle.load(open("coco/data/caption_id_2_img_id"))

        # load data to map from caption id to caption:
        if self.debug:
            self.train_caption_id_2_caption =\
                    cPickle.load(open("coco/data/val_caption_id_2_caption"))
        else:
            self.train_caption_id_2_caption =\
                    cPickle.load(open("coco/data/train_caption_id_2_caption"))

        # load data to map from img id to feature array:
        self.img_id_2_feature_array =\
                cPickle.load(open("coco/data/img_id_2_feature_array"))

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
                    shape=[None, self.config.max_caption_length], # [batch_size, max_caption_length]
                    name="captions_ph")
        # add the placeholder for the batch imgs (imgs_ph[i] will be the
        # img feature vectors for ex i in the batch):
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[None, self.config.no_of_img_feature_vecs,
                           self.config.img_feature_dim], # [batch_size, 64, 300]
                    name="imgs_ph")
        # add the placeholder for the batch labels (row i of labels_ph will
        # be the labels/targets for ex i in the batch):
        self.labels_ph = tf.placeholder(tf.int32,
                    shape=[None, self.config.max_caption_length], # [batch_size, max_caption_length]
                    name="labels_ph")
        # add the placeholder for the keep_prob (with what probability we will
        # keep a hidden unit during training):
        self.dropout_ph = tf.placeholder(tf.float32, name="dropout_ph")

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

    def add_captions_input(self):
        """
        - DOES: gets the word vector for each tokenized word in captions_ph
        giving a tensor of shape [batch_size, max_caption_length, embed_dim].
        """

        # get the word vector for each tokenized word in captions_ph:
        with tf.variable_scope("GRU_att_captions_embed"):
            # initialize the embeddings matrix with pretrained GloVe vectors (
            # note that we will train the embeddings matrix as well!):
            word_embeddings = tf.get_variable("word_embeddings",
                        initializer=self.GloVe_embeddings)
            # get the word vectors (gives a tensor of shape
            # [batch_size, max_caption_length, embed_dim]):
            self.captions_input = tf.nn.embedding_lookup(word_embeddings,
                        self.captions_ph)

    def add_logits(self):
        """
        - DOES: computes the network input at each timestep by concatenating the
        current word and an img feature vector (this feature vector is a sum of
        the img's 64 feature vectors scaled by an attention probability
        distribution, which is computed in an NN based on the previous hidden
        state), feeds each input through a GRU, producing a hidden state vector
        for each word/img and computes all corresponding logits (unnormalized
        prediction probabilties over the vocabulary, a softmax step but without
        the actual softmax).
        """

        # create a GRU cell:
        GRU_cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)
        # apply dropout to the GRU cell:
        GRU_cell = tf.nn.rnn_cell.DropoutWrapper(GRU_cell,
                    input_keep_prob=self.dropout_ph,
                    output_keep_prob=self.dropout_ph)
        # stack no_of_layers GRU cells on top of each other (for a deep GRU):
        stacked_GRU_cell = tf.nn.rnn_cell.MultiRNNCell(
                    [GRU_cell]*self.config.no_of_layers)
        # initialize the state of the stacked GRU cell (tf.shape(self.input)[0]
        # gets the current batch size) (the state contains both h and c for all
        # layers, thus its format is not trivial):
        initial_state = stacked_GRU_cell.zero_state(tf.shape(self.captions_input)[0],
                    tf.float32)

        with tf.variable_scope("GRU_attention"):
            # initialize the attention NN paramaters:
            W_a_h = tf.get_variable("W_a_h",
                        shape=[self.config.hidden_dim, self.config.hidden_dim_att],
                        initializer=tf.contrib.layers.xavier_initializer())
            W_a_I = tf.get_variable("W_a_I",
                        shape=[self.config.img_feature_dim, self.config.hidden_dim_att],
                        initializer=tf.contrib.layers.xavier_initializer())
            # (W is W_a in the paper)
            W = tf.get_variable("W",
                        shape=[self.config.hidden_dim_att, 1],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_a = tf.get_variable("b_a",
                        shape=[1, self.config.hidden_dim_att],
                        initializer=tf.constant_initializer(0))
            b_alpha = tf.get_variable("b_alpha",
                        shape=[1, self.config.no_of_img_feature_vecs],
                        initializer=tf.constant_initializer(0))

            # compute the network input for each timestep, feed it through the
            # stacked GRU cell, get the (top) hidden state vector and the
            # corresponding attention probabilities:
            attention_maps = []
            outputs = []
            previous_state = initial_state
            for timestep in range(self.config.max_caption_length):
                if timestep > 0:
                    # make sure we reuse variables in the stacked GRU cell:
                    tf.get_variable_scope().reuse_variables()

                # compute the attention probabilities:
                if timestep == 0:
                    # set att_probs so that the inital img feature vector is
                    # the average of the feature vectors:
                    att_probs = (1./self.config.no_of_img_feature_vecs)*tf.ones((
                                tf.shape(self.captions_input)[0],
                                self.config.no_of_img_feature_vecs, 1))
                    # (att_probs has shape [batch_size, 64, 1])
                else:
                    # compute att_probs (alpha) according to the paper:
                    previous_output = outputs[timestep-1] # (h_{t-1})

                    previous_output_trans = tf.matmul(previous_output, W_a_h)
                    # (previous_output_trans has shape [batch_size, hidden_dim_att])

                    ###### slow but intuitive way to compute a:
                    # a = []
                    # for i in range(self.config.no_of_img_feature_vecs):
                    #     z_i_linear = tf.matmul(self.imgs_ph[:, i, :], W_a_I) +\
                    #                 previous_output_trans + b_a
                    #     z_i = tf.nn.tanh(z_i_linear)
                    #     # (self.imgs_ph[:, i, :] has shape [batch_size, 300])
                    #     # (z_i has shape [batch_size, hidden_dim_att])
                    #
                    #     a_i = tf.matmul(z_i, W)
                    #     # (a_i has shape [batch_size, 1])
                    #     a.append(a_i)
                    ####################################

                    # ###### faster way to compute a:
                    x = tf.transpose(self.imgs_ph, [1, 0, 2])
                    # (imgs_ph has shape [batch_size, 64, 300])
                    # (x has shape [64, batch_size, 300])
                    x = tf.reshape(x, [-1, self.config.img_feature_dim])
                    # (x has shape [batch_size*64, 300])

                    x_trans = tf.matmul(x, W_a_I)
                    # (x_trans has shape [batch_size*64, hidden_dim_att])
                    previous_output_trans = tf.tile(previous_output_trans,
                                [self.config.no_of_img_feature_vecs, 1])
                    # (previous_output_trans has shape [batch_size*64, hidden_dim_att])

                    y = tf.nn.tanh(x_trans + previous_output_trans + b_a)
                    y = tf.nn.dropout(y, self.dropout_ph)
                    # (y has shape [batch_size*64, hidden_dim_att])

                    a = tf.matmul(y, W)
                    # (a has shape [batch_size*64, 1])
                    a = tf.split(0, self.config.no_of_img_feature_vecs, a)
                    ####################################

                    # (a is a list of 64 elements, each of which is a
                    # tensor of shape [batch_size, 1])
                    # reshape a into shape [batch_size, 64, 1]:
                    a = tf.pack(a, axis=1)
                    # reshape a into shape [batch_size, 64]:
                    a = tf.reshape(a, [tf.shape(self.captions_input)[0], self.config.no_of_img_feature_vecs])

                    # turn into probabilities using a softmax:
                    att_probs_linear = a + b_alpha
                    att_probs = tf.nn.softmax(att_probs_linear)
                    # (att_probs has shape [batch_size, 64])

                    att_probs = tf.expand_dims(att_probs, 2)
                    # (att_probs has shape [batch_size, 64, 1])

                # set the timestep's img feature vector to be a scaled sum
                # of all 64 feature vectors (scaled by att_probs):
                scaled_img_features = att_probs*self.imgs_ph
                step_imgs_input = tf.reduce_sum(scaled_img_features, axis=1)
                # (step_imgs_input has shape [batch_size, 300])

                # get the word (for every ex in the batch) corresponding to
                # the timestep:
                step_captions_input = self.captions_input[:, timestep, :]
                # (step_captions_input has shape [batch_size, 300])

                # concatenate the timestep's img input and caption input to
                # get the network input:
                step_input = tf.concat(1, [step_imgs_input, step_captions_input])
                # (step_input has shape [batch_size, 600])

                # get the (top) hidden state vector for the timestep:
                output, new_state = stacked_GRU_cell(step_input, previous_state)
                # (output = h (at the top layer), new_state is a tuple containing
                # both h and c (for all layers if we have more than one layer),
                # new_state is thus needed to compute the output at the next timestep)
                # (output thus has shape [batch_size, hidden_dim])

                # save new_state (will be used to compute output in the next timestep):
                previous_state = new_state

                # add the timestep's output:
                outputs.append(output)

                # add the timestep's att_probs:
                attention_maps.append(att_probs)

        # save the attention probs to enable visualization:
        self.attention_maps = attention_maps

        # (outputs is a list of max_caption_length elements, each of which is a
        # tensor of shape [batch_size, hidden_dim])
        # reshape outputs into shape [batch_size, max_caption_length, hidden_dim]:
        outputs = tf.pack(outputs, axis=1)
        # reshape outputs into shape [batch_size*max_caption_length, hidden_dim]:
        outputs = tf.reshape(outputs, [-1, self.config.hidden_dim])

        # compute corresponding logits for each hidden state vector in outputs,
        # resulting in a tensor self.logits of shape
        # [batch_size*max_caption_length, vocab_size] (each word in the batch
        # will have a corr. logits vector, which is an unnorm. prob. distr. over
        # the vocab. The largets element corresponds to the predicted next word):
        with tf.variable_scope("GRU_att_logits"):
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

        # reshape labels_ph into shape [batch_size*max_caption_length, ] (to
        # match the shape of self.logits):
        labels = tf.reshape(self.labels_ph, [-1])

        # remove all -1 labels and their corresponding logits (-1 labels
        # correspond to the <EOS> step or padded steps, the predicitons at these
        # steps are irrelevant and should not contribute to the loss):
        mask = tf.greater_equal(labels, 0)
        masked_labels = tf.boolean_mask(labels, mask)
        masked_logits = tf.boolean_mask(self.logits, mask)

        # compute the CE loss for each word in the batch:
        loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    masked_logits, masked_labels)
        # average the loss over all words to get the batch loss:
        loss = tf.reduce_mean(loss_per_word)

        self.loss = loss

    def add_training_op(self):
        """
        - DOES: creates a training operator for optimizing the loss.
        """

        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr)
        self.train_op = optimizer.minimize(self.loss)

    def run_epoch(self, session):
        """
        - DOES: runs one epoch, i.e., for each batch it: computes the batch loss
        (forwardprop), computes all gradients w.r.t to the batch loss and updates
        all network variables/parameters accordingly (backprop).
        """

        batch_losses = []
        for step, (captions, imgs, labels) in enumerate(train_data_iterator_attention(self)):
            # create a feed_dict with the batch data:
            feed_dict = self.create_feed_dict(captions, imgs,
                        labels_batch=labels, dropout=self.config.dropout)
            # compute the batch loss and compute & apply all gradients w.r.t to
            # the batch loss (without self.train_op in the call, the network
            # would not train, we would only compute the batch loss):
            batch_loss, _ = session.run([self.loss, self.train_op],
                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)

            if step % 100 == 0:
                print "batch: %d | loss: %f" % (step, batch_loss)
                log("batch: %d | loss: %f" % (step, batch_loss))

            if step > 2 and self.debug:
                break

        # return a list containing the batch loss for each batch:
        return batch_losses

    def generate_img_caption(self, session, img_features, vocabulary):
        """
        - DOES: generates a caption for the img feature vectors img_features.
        If mode == "demo", it also returns the attention probabilities for all
        words, allowing visualization of the attention mechanism.
        """

        # initialize the caption as "<SOS>":
        caption = np.zeros((1, self.config.max_caption_length))
        caption[0, 0] = np.array(vocabulary.index("<SOS>"))
        # format the img_vector so it can be fed to the NN:
        img = np.zeros((1, self.config.no_of_img_feature_vecs,
                    self.config.img_feature_dim))
        img[0] = img_features

        # we will get one vector of logits for each timestep, element 0 corr. to
        # <SOS>, element 1 corr. to the first word etc., to begin we want to get
        # the one corr. to "<SOS>":
        prediction_index = 0

        # predict the next word until we get "<EOS>" or the caption length hits
        # a max value:
        attention_maps = []
        while int(caption[0][prediction_index]) is not vocabulary.index("<EOS>") and\
                    prediction_index < self.config.max_caption_length-1:
            feed_dict = self.create_feed_dict(caption, img)

            if self.mode == "demo":
                # get both the logits and the attention probs (for visualization):
                logits, att_maps = session.run([self.logits, self.attention_maps],
                            feed_dict=feed_dict)
            else:
                # get only the logits:
                logits = session.run(self.logits, feed_dict=feed_dict)

            # get the logits vector corr. to the last word in the current caption
            # (it gives what next word we will predict):
            prediction_logits = logits[prediction_index]
            # get the index of the predicted word (the word in the vocabulary
            # with the largest (unnormalized) probability)
            predicted_word_index = np.argmax(prediction_logits)
            # add the new word to the caption:
            caption[0, prediction_index+1] = predicted_word_index

            if self.mode == "demo":
                # get the attention probs corr. to the predicted word:
                attention_probs = att_maps[prediction_index]
                attention_maps.append(attention_probs)

            # increment prediction_index so that we'll look at the new last word
            # of the caption in the next iteration:
            prediction_index += 1

        # get the caption and convert to ints:
        caption = caption[0].astype(int)
        # remove all padding:
        caption = caption[0:prediction_index+1]
        # convert the caption to actual text:
        caption = detokenize_caption(caption, vocabulary)

        if self.mode == "demo":
            # remove the attention_probs corr to the prediction when at the
            # timestep of <EOS> (this is irrelevant):
            attention_maps.pop()

        if self.mode == "demo":
            return caption, attention_maps
        else:
            return caption

    def generate_captions_on_val(self, session, epoch, vocabulary, val_set_size=5000):
        """
        - DOES: generates a caption for each of the first val_set_size imgs in
        the val set, saves them in the format expected by the provided COCO
        evaluation script and returns the name of the saved file.
        """

        if self.debug:
            val_set_size = 5

        # get the img ids of all val imgs:
        val_img_ids = cPickle.load(open("coco/data/val_img_ids"))
        # take the first val_set_size val imgs:
        val_set = val_img_ids[0:val_set_size]

        captions = []
        for step, img_id in enumerate(val_set):
            if step % 100 == 0:
                print "generating captions on val: %d" % step
                log("generating captions on val: %d" % step)

            # get the img's img feature vectors from disk:
            #img_features = cPickle.load(
            #            open("coco/data/img_features_attention/%d" % img_id))
            img_features = self.img_id_2_feature_array[img_id]

            # generate a caption for the img:
            img_caption = self.generate_img_caption(session, img_features, vocabulary)
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
    # create a config object:
    config = GRU_attention_Config()
    # get the pretrained embeddings matrix:
    GloVe_embeddings = cPickle.load(open("coco/data/embeddings_matrix"))
    GloVe_embeddings = GloVe_embeddings.astype(np.float32)
    # create a GRU_attention model object:
    model = GRU_attention_Model(config, GloVe_embeddings)

    # initialize the list that will contain the loss for each epoch:
    loss_per_epoch = []
    # initialize the list that will contain all evaluation metrics (BLEU, CIDEr,
    # METEOR and ROUGE_L) for each epoch:
    eval_metrics_per_epoch = []

    # create a saver for saving all model variables/parameters:
    saver = tf.train.Saver(max_to_keep=model.config.max_no_of_epochs)

    with tf.Session() as sess:
        # initialize all variables/parameters:
        init = tf.global_variables_initializer()
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

            # run an epoch and get all batch losses:
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
            # evaluate the generated captions (compute eval metrics):
            eval_result_dict = evaluate_captions(captions_file)
            # save the epoch evaluation metrics:
            eval_metrics_per_epoch.append(eval_result_dict)
            # save the evaluation metrics for all epochs to disk:
            cPickle.dump(eval_metrics_per_epoch, open("%s/eval_results/metrics_per_epoch"\
                        % model.config.model_dir, "w"))

            if eval_result_dict["CIDEr"] > 0.85:
                # save the model weights to disk:
                saver.save(sess, "%s/weights/model" % model.config.model_dir,
                            global_step=epoch)

            print "epoch loss: %f | BLEU4: %f  |  CIDEr: %f" % (epoch_loss,
                        eval_result_dict["Bleu_4"], eval_result_dict["CIDEr"])
            log("epoch loss: %f | BLEU4: %f  |  CIDEr: %f" % (epoch_loss,
                        eval_result_dict["Bleu_4"], eval_result_dict["CIDEr"]))

    # plot the loss and the different metrics vs epoch:
    plot_performance(config.model_dir)

if __name__ == '__main__':
    main()
