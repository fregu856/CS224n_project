import numpy as np
import tensorflow as tf

import cPickle
import os
import time
import json
import cPickle
import random

from utilities import train_data_iterator, detokenize_caption, evaluate_captions
from utilities import plot_performance, compare_captions

class Config(object):

    def __init__(self):
        self.dropout = 0.5
        self.embed_dim = 300
        self.hidden_dim = 200
        self.batch_size = 256
        self.lr = 0.001
        self.img_dim = 2048
        self.vocab_size = 9855
        self.no_of_layers = 1
        self.max_no_of_epochs = 50
        self.max_caption_length = 30
        self.model_name = "model_keep=%.2f_batch=%d_hidden_dim=%d_embed_dim=%d_layers=%d" % (self.dropout,
                    self.batch_size, self.hidden_dim, self.embed_dim,
                    self.no_of_layers)
        self.model_dir = "models/LSTMs/%s" % self.model_name

class Model(object):

    def __init__(self, config, GloVe_embeddings, debug=False, mode="training"):
        self.GloVe_embeddings = GloVe_embeddings
        self.debug = debug
        self.config = config
        if mode is not "demo":
            self.create_model_dirs()
            self.load_utilities_data()
        self.add_placeholders()
        self.add_input()
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

    def add_placeholders(self):
        self.captions_ph = tf.placeholder(tf.int32,
                    shape=[self.config.batch_size, None],
                    name="captions_ph")
        self.imgs_ph = tf.placeholder(tf.float32,
                    shape=[self.config.batch_size, self.config.img_dim],
                    name="imgs_ph")
        self.labels_ph = tf.placeholder(tf.int32,
                    shape=[self.config.batch_size, None],
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

    def add_input(self):
        with tf.variable_scope("img_transform"):
            W_img = tf.get_variable("W_img",
                        shape=[self.config.img_dim, self.config.embed_dim],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_img = tf.get_variable("b_img", shape=[1, self.config.embed_dim],
                        initializer=tf.constant_initializer(0))
            imgs_input = tf.nn.sigmoid(tf.matmul(self.imgs_ph, W_img) + b_img)
            imgs_input = tf.expand_dims(imgs_input, 1)

        with tf.variable_scope("captions_embed"):
            word_embeddings = tf.get_variable("word_embeddings",
                        initializer=self.GloVe_embeddings)
            captions_input = tf.nn.embedding_lookup(word_embeddings,
                        self.captions_ph)

        self.input = tf.concat(1, [imgs_input, captions_input])

    def add_logits(self):
        LSTM = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_dim)
        LSTM_dropout = tf.nn.rnn_cell.DropoutWrapper(LSTM,
                    input_keep_prob=self.dropout_ph,
                    output_keep_prob=self.dropout_ph)
        multilayer_LSTM = tf.nn.rnn_cell.MultiRNNCell(
                    [LSTM_dropout]*self.config.no_of_layers)
        initial_state = multilayer_LSTM.zero_state(self.config.batch_size,
                    tf.float32)
        outputs, final_state = tf.nn.dynamic_rnn(multilayer_LSTM,
                    self.input, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, self.config.hidden_dim])

        with tf.variable_scope("logits"):
            W_logits = tf.get_variable("W_logits",
                        shape=[self.config.hidden_dim, self.config.vocab_size],
                        initializer=tf.contrib.layers.xavier_initializer())
            b_logits = tf.get_variable("b_logits",
                        shape=[1, self.config.vocab_size],
                        initializer=tf.constant_initializer(0))
            self.logits = tf.matmul(output, W_logits) + b_logits

    def add_loss_op(self):
        labels = tf.reshape(self.labels_ph, [-1])

        # remove all -1 labels and their corresponding logits (-1 labels
        # correspond to the img or <EOS> step, the predicitons at these
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

        for step, (captions, imgs, labels) in enumerate(train_data_iterator(self)):
            feed_dict = self.create_feed_dict(captions, imgs,
                        labels_batch=labels, dropout=self.config.dropout)
            batch_loss, _ = session.run([self.loss, self.train_op],
                        feed_dict=feed_dict)
            batch_losses.append(batch_loss)

            if step % 10 == 0:
                print "batch: %d | loss: %f" % (step, batch_loss)

            if step > 4 and self.debug:
                break

        return batch_losses

    def generate_img_caption(self, session, img_vector, vocabulary):
        # (the NN always needs to be fed tensors of shape (batch_size, ?), but
        # the only thing we care about here is the first row)

        # initialize the caption as "<SOS>":
        caption = np.zeros((self.config.batch_size, 1))
        caption[0] = np.array(vocabulary.index("<SOS>"))
        # format the img_vector so it can be fed to the NN:
        img = np.zeros((self.config.batch_size, self.config.img_dim))
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
            new_word_col = np.zeros((self.config.batch_size, 1))
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
            val_set_size = 2

        # get the map from img id to feature vector:
        val_img_id_2_feature_vector =\
                    cPickle.load(open("coco/data/val_img_id_2_feature_vector"))
        # turn the map into a list of tuples (to make it iterable):
        val_img_id_feature_vector_list = val_img_id_2_feature_vector.items()\
        # randomly shuffle the list of tuples (to take different subsets when
        # val_set_size is not set to 5000):
        random.shuffle(val_img_id_feature_vector_list)
        # take a subset (of size val_set_size) of all val imgs:
        val_set = val_img_id_feature_vector_list[0:val_set_size]

        captions = []
        for step, (img_id, img_vector) in enumerate(val_set):
            if step % 10 == 0:
                print "generating captions on val: %d" % step

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

def main(debug=False):
    config = Config()
    GloVe_embeddings = cPickle.load(open("coco/data/embeddings_matrix"))
    GloVe_embeddings = GloVe_embeddings.astype(np.float32)
    model = Model(config, GloVe_embeddings)

    loss_per_epoch = []
    eval_metrics_per_epoch = []

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(config.max_no_of_epochs):
            print "###########################"
            print "######## NEW EPOCH ########"
            print "###########################"
            print "epoch: %d/%d" % (epoch, config.max_no_of_epochs-1)

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
                        model.vocabulary, val_set_size=25)
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

            print "epoch loss: %f | BLEU4: %f" % (epoch_loss, eval_result_dict["Bleu_4"])

    # plot the loss and the different metrics vs epoch:
    plot_performance(config.model_dir)

    #compare_captions(config.model_dir, 3)

if __name__ == '__main__':
    main()
