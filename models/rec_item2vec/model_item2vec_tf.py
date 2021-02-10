# Author: Hayden Lao
# Script Name: model_item2vec_tf
# Created Date: Apr 12th 2020
# Description: Item2Vec model built by tensorflow for movieLens recommendation

import collections
import numpy as np
import tensorflow as tf
from functools import reduce
from tensorflow.contrib.lookup import index_table_from_tensor


class Item2VecTF:
    def __init__(self, **params):
        self.act_seq_file = params["act_seq_file"]  # action sequence file
        self.act_seq_separator = params["act_seq_separator"]  # action sequence separator
        self.learning_rate = params["learning_rate"]  # learning rate
        self.batch_size = params["batch_size"]  # batch size
        self.emb_size = params["emb_size"]  # vector dimension
        self.window = params["window"]  # slide window
        self.min_count = params["min_count"]  # minimum action number
        self.network_algo = params["network_algo"]  # network algorithm: 0 for CBOW, 1 for skip-gram
        self.speed_algo = params["speed_algo"]  # speed algorithm: 0 for negative sampling, 1 for hierarchical softmax
        self.negative_number = params["negative_number"]  # negative sampling number

    def read_seq_file(self):
        seq_list = []
        # flatten action sequence
        with open(self.act_seq_file) as f:
            for line in f:
                seq_list.append(line.strip().split(self.act_seq_separator))
        return seq_list

    def make_item2id(self, seq_list):
        count_dict = collections.Counter(reduce(lambda seq1, seq2: seq1 + seq2, seq_list))
        item_mapper = list(map(lambda i: i[0] if i[1] >= self.min_count else '0', count_dict.items()))
        item2id = index_table_from_tensor(mapping=item_mapper, default_value=0)
        self.vocab_size = item2id.size()
        return item2id

    def gen_batch(self, seq_list, item2id):
        batch_inputs, batch_labels = [], []
        # handle each sequence
        for seq in seq_list:
            # handle each item
            for seq_idx in range(len(seq)):
                # prevent out of bounds
                start = max(0, seq_idx - self.window)
                end = min(len(seq), seq_idx + self.window)
                for window_idx in range(start, end):
                    if window_idx == seq_idx:
                        continue
                    batch_inputs.append(item2id.lookup(seq[seq_idx]))
                    batch_labels.append(item2id.lookup(seq[window_idx]))
        if len(batch_inputs) == 0:
            return
        batch_inputs = np.array(batch_inputs, dtype=np.int32)
        batch_labels = np.array(batch_labels, dtype=np.int32).reshape([len(batch_labels), 1])
        return batch_inputs, batch_labels

    def build_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.name_scope("inputs"):
                train_inputs = tf.placeholder(tf.int32, shape=[self.batch_size])
                train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            with tf.device("/cpu:0"):
                with tf.variable_scope("embeddings"):
                    embeddings = tf.Variable(
                        tf.random_uniform(shape=[self.vocab_size, self.emb_size], minval=-1.0, maxval=1.0))
                    emb = tf.nn.embedding_lookup(embeddings, train_inputs)
                with tf.variable_scope("weights"):
                    nce_weights = tf.Variable(tf.truncated_normal(shape=[self.vocab_size, self.emb_size], stddev=0.01))
                with tf.variable_scope("biases"):
                    nce_biases = tf.Variable(tf.zeros[self.vocab_size])
                with tf.name_scope("nce_loss"):
                    nce_loss = tf.reduce_mean(
                        tf.nn.nce_loss(
                            weights=nce_weights,
                            biases=nce_biases,
                            labels=train_labels,
                            inputs=train_inputs, num_sampled=self.negative_number, num_classes=self.vocab_size
                        )
                    )
                tf.summary.scalar("nce_loss", nce_loss)

                with tf.name_scope("optimizer"):
                    optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(nce_loss)
