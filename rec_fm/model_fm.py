import tensorflow as tf

from rec_layers.layer_vocab import *
from rec_layers.layer_embedding import *


class FM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.factor_dim = param_dict["factor_dim"]
        self.vocab_list = param_dict["vocab_list"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.embedding_layer1 = EmbeddingLayer(feature_size=self.feature_size, emb_size=1)
        self.embedding_layer2 = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.factor_dim)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        all_inputs = tf.concat(list(inputs.values()), axis=1)
        # vocab layer
        self.vocab = self.vocab_layer(all_inputs)
        # global bias
        self.global_bias = tf.keras.initializers.constant(0.0)([1])
        # first order
        self.first_weights = self.embedding_layer1(self.vocab)
        self.first_weights = tf.squeeze(self.first_weights, axis=2)
        self.first_score = tf.reduce_sum(self.first_weights, axis=1)
        # second_order
        self.second_weights = self.embedding_layer2(self.vocab)
        self.sum_square = tf.square(tf.reduce_sum(self.second_weights, axis=1))
        self.square_sum = tf.reduce_sum(tf.square(self.second_weights), axis=1)
        self.second_score = 0.5 * tf.reduce_sum((self.sum_square - self.square_sum), axis=1)
        # fm score
        self.score = self.global_bias + self.first_score + self.second_score
        self.outputs = self.out_layer(self.score)
        return self.outputs

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "factor_dim": self.factor_dim,
                       "vocab_list": self.vocab_list, })
        return config
