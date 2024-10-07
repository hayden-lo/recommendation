from layers.embedding_layer import *
from layers.mlp_layer import *
from layers.vocab_layer import *


class DeepFM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.vocab_list = param_dict["vocab_list"]
        self.act_fun = param_dict["act_fun"]
        self.reg_fun = param_dict["reg_fun"]
        self.hidden_units = param_dict["hidden_units"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size)
        self.dense_layer = DenseLayer(1, act_fun=self.act_fun, reg_fun=self.reg_fun)
        self.mlp_layer = MLPLayer(hidden_units=self.hidden_units, act_fun=self.act_fun, reg_fun=self.reg_fun)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        all_inputs = tf.concat(list(inputs.values()), axis=1)
        # vocab layer
        self.vocab = self.vocab_layer(all_inputs)
        # embedding layer
        self.embedding = self.embedding_layer(self.vocab)
        # fm
        self.global_bias = tf.keras.initializers.constant(0.0)([1])
        self.first_weights = self.dense_layer(self.embedding)
        self.first_weights = tf.squeeze(self.first_weights, axis=2)
        self.first_score = tf.reduce_sum(self.first_weights, axis=1)
        self.sum_square = tf.square(tf.reduce_sum(self.embedding, axis=1))
        self.square_sum = tf.reduce_sum(tf.square(self.embedding), axis=1)
        self.second_score = 0.5 * tf.reduce_sum((self.sum_square - self.square_sum), axis=1)
        self.fm_score = self.global_bias + self.first_score + self.second_score
        self.fm_score = tf.expand_dims(self.fm_score, axis=1)
        # deep
        self.deep_inputs = tf.reshape(self.embedding, shape=[-1, all_inputs.shape[1] * self.emb_size])
        self.deep_score = self.mlp_layer(self.deep_inputs)
        # deepfm score
        self.deepfm_score = self.fm_score + self.deep_score
        self.outputs = self.out_layer(self.deepfm_score)
        return self.outputs

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "vocab_list": self.vocab_list,
                       "act_fun": self.act_fun, "reg_fun": self.reg_fun, "hidden_units": self.hidden_units})
        return config
