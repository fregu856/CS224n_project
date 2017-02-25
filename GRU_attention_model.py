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

    def __init__(self, debug=False):
        self.dropout = 0.5
        self.embed_dim = 300
        self.hidden_dim = 200
        self.batch_size = 256
        self.lr = 0.001
        self.img_feature_dim = 300
        self.no_of_img_feature_vecs = 64
        self.vocab_size = 9855
        self.no_of_layers = 1
        if debug:
            self.max_no_of_epochs = 2
        else:
            self.max_no_of_epochs = 60
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim, self.embed_dim,
                    self.no_of_layers)
        self.model_dir = "models/GRUs_attention/%s" % self.model_name
        self.max_caption_length = get_max_caption_length(self.batch_size)
        self.hidden_dim_att = 200

class GRU_attention_Model(object):

    def __init__(self, config, GloVe_embeddings, debug=False, mode="training"):
        self.GloVe_embeddings = GloVe_embeddings
        self.debug = debug
        self.config = config
        self.mode = mode
        if mode is not "demo":
            self.create_model_dirs()
            self.load_utilities_data()
        self.add_placeholders()
        self.add_captions_input()
        self.add_logits()
        if mode is not "demo":
            self.add_loss_op()
            self.add_training_op()

    def create_model_dirs(self):
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

        # create the dir where performance plots will be saved during training:
        if not os.path.exists("%s/plots" % self.config.model_dir):
            os.mkdir("%s/plots" % self.config.model_dir)

    def load_utilities_data(self):
        print "loading utilities data..."
        log("loading utilities data...")

        # load the vocabulary:
        self.vocabulary = cPickle.load(open("coco/data/vocabulary"))

        # load data to map from caption id to img_id:
        self.caption_id_2_img_id =\
                    cPickle.load(open("coco/data/caption_id_2_img_id"))

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
        self.captions_ph = tf.placeholder(tf.int32,
                    shape=[None, self.config.max_caption_length], # [batch_size, max_caption_length]
                    name="captions_ph")
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[None, self.config.no_of_img_feature_vecs,
                           self.config.img_feature_dim], # [batch_size, 64, 300]
                    name="imgs_ph")
        self.labels_ph = tf.placeholder(tf.int32,
                    shape=[None, self.config.max_caption_length], # [batch_size, max_caption_length]
                    name="labels_ph")
        self.dropout_ph = tf.placeholder(tf.float32, name="dropout_ph")

    def create_feed_dict(self, captions_batch, imgs_batch, labels_batch=None, dropout=1):
        feed_dict = {}
        feed_dict[self.captions_ph] = captions_batch
        feed_dict[self.imgs_ph] = imgs_batch
        feed_dict[self.dropout_ph] = dropout
        if labels_batch is not None:
            feed_dict[self.labels_ph] = labels_batch

        return feed_dict

    def add_captions_input(self):
        with tf.variable_scope("GRU_att_captions_embed"):
            word_embeddings = tf.get_variable("word_embeddings",
                        initializer=self.GloVe_embeddings)
            self.captions_input = tf.nn.embedding_lookup(word_embeddings,
                        self.captions_ph)
            # (captions_input has shape [batch_size, max_caption_length, 300])

    def add_logits(self):
        GRU = tf.nn.rnn_cell.GRUCell(self.config.hidden_dim)
        GRU_dropout = tf.nn.rnn_cell.DropoutWrapper(GRU,
                    input_keep_prob=self.dropout_ph,
                    output_keep_prob=self.dropout_ph)
        multilayer_GRU = tf.nn.rnn_cell.MultiRNNCell(
                    [GRU_dropout]*self.config.no_of_layers)
        initial_state = multilayer_GRU.zero_state(tf.shape(self.captions_input)[0],
                    tf.float32)
        # (tf.shape(self.captions_input)[0] gets the current batch size)

        attention_maps = []
        outputs = []
        previous_state = initial_state
        with tf.variable_scope("GRU_attention"):
            W_att = tf.get_variable("W_att",
                        shape=[self.config.hidden_dim, self.config.hidden_dim_att],
                        initializer=tf.contrib.layers.xavier_initializer())
            U_att = tf.get_variable("U_att",
                        shape=[self.config.hidden_dim_att, self.config.no_of_img_feature_vecs],
                        initializer=tf.contrib.layers.xavier_initializer())
            b1_att = tf.get_variable("b1_att",
                        shape=[1, self.config.hidden_dim_att],
                        initializer=tf.constant_initializer(0))
            b2_att = tf.get_variable("b2_att",
                        shape=[1, self.config.no_of_img_feature_vecs],
                        initializer=tf.constant_initializer(0))

            for timestep in range(self.config.max_caption_length):
                if timestep > 0:
                    # make sure we reuse variables:
                    tf.get_variable_scope().reuse_variables()

                if timestep == 0:
                    # set att_probs so that the inital img feature vector is
                    # the average of the feature vectors:
                    att_probs = (1./self.config.no_of_img_feature_vecs)*tf.ones((
                                tf.shape(self.captions_input)[0],
                                self.config.no_of_img_feature_vecs, 1))
                    # (att_probs has shape [batch_size, 64, 1])
                else:
                    # compute att_probs with a one hidden layer NN:
                    previous_output = outputs[timestep-1] # (h_{t-1})
                    h_att_linear = tf.matmul(previous_output, W_att) + b1_att
                    h_att = tf.nn.relu(h_att_linear)
                    # (h_att has shape [batch_size, hidden_dim_att])

                    # turn into probabilities using a softmax:
                    att_probs_linear = tf.matmul(h_att, U_att) + b2_att
                    att_probs = tf.nn.softmax(att_probs_linear)
                    # (att_probs has shape [batch_size, 64])

                    att_probs = tf.expand_dims(att_probs, 2)
                    # (att_probs has shape [batch_size, 64, 1])

                # set the timestep's img feature vector to be a scaled sum
                # of all 64 feature vectors (scaled by att_probs):
                scaled_img_features = att_probs*self.imgs_ph
                step_imgs_input = tf.reduce_sum(scaled_img_features, axis=1)
                # (step_imgs_input has shape [batch_size, 300])

                # get the word (for entire batch) corresponding to the timestep:
                step_captions_input = self.captions_input[:, timestep, :]
                # (step_captions_input has shape [batch_size, 300])

                step_input = tf.concat(1, [step_imgs_input, step_captions_input])
                # (step_input has shape [batch_size, 600])

                output, new_state = multilayer_GRU(step_input, previous_state)
                # (output = h (at the top layer), new_state is a tuple containing
                # both h and c (for all layers if we have more than one layer))
                # (output thus has shape [batch_size, hidden_dim])

                previous_state = new_state
                outputs.append(output)

                # save the attention probs to enable visualization:
                attention_maps.append(att_probs)

        self.attention_maps = attention_maps

        # (outputs is a list of max_caption_length elements, each of which is a
        # tensor of shape [batch_size, hidden_dim]. We first want a tensor of
        # shape [batch_size, max_caption_length, hidden_dim])
        outputs = tf.pack(outputs, axis=1)

        outputs = tf.reshape(outputs, [-1, self.config.hidden_dim])
        # (outputs has shape [batch_size*max_caption_length, hidden_dim])

        with tf.variable_scope("GRU_att_logits"):
            W_logits = tf.get_variable("W_logits",
                        shape=[self.config.hidden_dim, self.config.vocab_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_logits = tf.get_variable("b_logits",
                        shape=[1, self.config.vocab_size],
                        initializer=tf.constant_initializer(0))
            self.logits = tf.matmul(outputs, W_logits) + b_logits
            # (self.logits has shape [batch_size*max_caption_length, vocab_size])

    def add_loss_op(self):
        labels = tf.reshape(self.labels_ph, [-1])
        # (labels_ph has shape [batch_size, max_caption_length])
        # (labels has shape [batch_size*max_caption_length, ])

        # remove all -1 labels and their corresponding logits (-1 labels
        # correspond to the <EOS> step or padded steps, the predicitons at these
        # steps are irrelevant):
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
        start_time = time.time()

        for step, (captions, imgs, labels) in enumerate(train_data_iterator_attention(self)):
            feed_dict = self.create_feed_dict(captions, imgs,
                        labels_batch=labels, dropout=self.config.dropout)
            batch_loss, _ = session.run([self.loss, self.train_op],
                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)

            if step % 100 == 0:
                print "batch: %d | loss: %f" % (step, batch_loss)
                log("batch: %d | loss: %f" % (step, batch_loss))

            if step > 2 and self.debug:
                break

        return batch_losses

    def generate_img_caption(self, session, img_features, vocabulary):
        # initialize the caption as "<SOS>":
        caption = np.zeros((1, self.config.max_caption_length))
        caption[0, 0] = np.array(vocabulary.index("<SOS>"))
        # format the img_vector so it can be fed to the NN:
        img = np.zeros((1, self.config.no_of_img_feature_vecs,
                    self.config.img_feature_dim))
        img[0] = img_features

        prediction_index = 0
        attention_maps = []

        # predict the next word until we get "<EOS>" or the caption length hits
        # a max value:
        while int(caption[0][prediction_index]) is not vocabulary.index("<EOS>") and\
                    prediction_index < self.config.max_caption_length-1:
            feed_dict = self.create_feed_dict(caption, img)

            if self.mode == "demo":
                logits, att_maps = session.run([self.logits, self.attention_maps],
                            feed_dict=feed_dict)
            else:
                logits = session.run(self.logits, feed_dict=feed_dict)

            # get the logits vector corr. to the predicted word:
            prediction_logits = logits[prediction_index]
            # get the vocabulary index of the predicted word:
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
            # remove the attention_probs corr to the <EOS> prediction:
            attention_maps.pop()

        if self.mode == "demo":
            return caption, attention_maps
        else:
            return caption

    def generate_captions_on_val(self, session, epoch, vocabulary, val_set_size=5000):
        if self.debug:
            val_set_size = 5

        val_img_ids = cPickle.load(open("coco/data/val_img_ids"))
        # randomly shuffle the img ids (to take different subsets when
        # val_set_size is not set to 5000):
        #random.shuffle(val_img_ids)
        # take a subset (of size val_set_size) of all val imgs:
        val_set = val_img_ids[0:val_set_size]

        captions = []
        for step, img_id in enumerate(val_set):
            if step % 100 == 0:
                print "generating captions on val: %d" % step
                log("generating captions on val: %d" % step)

            img_features = cPickle.load(
                        open("coco/data/img_features_attention/%d" % img_id))

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
    config = GRU_attention_Config(debug=True)
    GloVe_embeddings = cPickle.load(open("coco/data/embeddings_matrix"))
    GloVe_embeddings = GloVe_embeddings.astype(np.float32)
    model = GRU_attention_Model(config, GloVe_embeddings, debug=True)

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

    #compare_captions(config.model_dir, 7)

if __name__ == '__main__':
    main()
