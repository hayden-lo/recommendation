# Author: Hayden Lao
# Script Name: model_fm
# Created Date: Jul 26th 2020
# Description: Factorization machine model for movieLens recData recommendation

import tensorflow as tf


class FM:
    def __init__(self, features, param_dict):
        self.target_id = features["target_id"]
        self.cat_feats = features["cat_feats"]
        self.seq_feats = features["seq_feats"]
        self.cat_size = param_dict["cat_size"]
        self.seq_size = param_dict["seq_size"]
        self.factor_dim = param_dict["factor_dim"]
        self.max_seq_num = param_dict["max_seq_num"]
        self.feature_size = param_dict["feature_size"]
        self.learning_rate = param_dict["learning_rate"]
        self.normal_mean = param_dict["normal_mean"]
        self.normal_stddev = param_dict["normal_stddev"]
        self._build_modle()

    # initialize parameters
    def _initialize_weights(self):
        self.global_bias = tf.Variable(tf.constant(0.0, shape=[1], name="global_bias"))
        self.first_weights = tf.Variable(
            tf.random_normal(shape=[self.feature_size, 1], mean=self.normal_mean, stddev=self.normal_stddev,
                             name="first_weights"))
        self.second_weights = tf.Variable(
            tf.random_normal(shape=[self.feature_size, self.factor_dim], mean=self.normal_mean,
                             stddev=self.normal_stddev, name="second_weights"))

    def _build_modle(self):
        self.graph = tf.Graph()
        with self.graph.as_default() and tf.variable_scope(name_or_scope="fm", reuse=tf.AUTO_REUSE):
            self._build_fm()

    def _build_fm(self):
        self._initialize_weights()
        # first order
        target_emb1 = tf.nn.embedding_lookup(self.first_weights, self.target_id, name="target_emb1")
        cat_emb1 = tf.nn.embedding_lookup(self.first_weights, self.cat_feats, name="cat_emb1")
        cat_emb1 = tf.reshape(cat_emb1, shape=[-1, self.cat_size])
        seq_emb1 = tf.nn.embedding_lookup(self.first_weights, self.seq_feats, name="seq_emb1")
        seq_emb1 = tf.reshape(seq_emb1, shape=[-1, self.seq_size * self.max_seq_num])
        self.first_emb = tf.concat([target_emb1, cat_emb1, seq_emb1], axis=1)
        self.first_score = tf.reduce_sum(self.first_emb, axis=1)

        # second order
        target_emb2 = tf.nn.embedding_lookup(self.second_weights, self.target_id, name="target_emb2")
        target_emb2 = tf.reshape(target_emb2, shape=[-1, 1, self.factor_dim])
        cat_emb2 = tf.nn.embedding_lookup(self.second_weights, self.cat_feats, name="cat_emb2")
        cat_emb2 = tf.reshape(cat_emb2, shape=[-1, self.cat_size, self.factor_dim])
        seq_emb2 = tf.nn.embedding_lookup(self.second_weights, self.seq_feats, name="seq_emb2")
        seq_emb2 = tf.reshape(seq_emb2, shape=[-1, self.seq_size * self.max_seq_num, self.factor_dim])
        self.second_emb = tf.concat([target_emb2, cat_emb2, seq_emb2], axis=1)
        sum_square = tf.square(tf.reduce_sum(self.second_emb, axis=1))
        square_sum = tf.reduce_sum(tf.square(self.second_emb), axis=1)
        self.second_score = 0.5 * tf.reduce_sum(sum_square - square_sum, axis=1)
        sum_square = tf.square(tf.reduce_sum(self.second_emb, axis=1))
        square_sum = tf.reduce_sum(tf.square(self.second_emb), axis=1)

        # fm score
        self.fm_score = self.global_bias + self.first_score + self.second_score
        self.predictions = tf.sigmoid(self.fm_score)

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op

    def get_loss(self, labels):
        reg_loss = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        normal_loss = tf.reduce_sum(tf.losses.log_loss(predictions=self.predictions, labels=labels))
        return reg_loss + normal_loss
