from layers.attention_layer import *
from layers.embedding_layer import *
from layers.mlp_layer import *
from layers.vocab_layer import *


class DIN(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.vocab_list = param_dict["vocab_list"]
        self.act_fun = param_dict["act_fun"]
        self.reg_fun = param_dict["reg_fun"]
        self.hidden_units = param_dict["hidden_units"]
        self.padding_value = param_dict["padding_value"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size,
                                              reg_fun=self.reg_fun)
        self.attention_layer = AttentionLayer(param_dict)
        self.mlp_layer = MLPLayer(hidden_units=self.hidden_units, act_fun=self.act_fun, reg_fun=self.reg_fun)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        query_inputs = inputs["movieId"]
        keys_inputs = inputs["click_seq"]
        normal_inputs = tf.concat([v for k, v in inputs.items() if k not in ["movieId", "click_seq"]], axis=1)
        # vocab layer
        self.query_vocab = self.vocab_layer(query_inputs)
        self.keys_vocab = self.vocab_layer(keys_inputs)
        self.normal_vocab = self.vocab_layer(normal_inputs)
        # embedding layer
        self.query_embedding = self.embedding_layer(self.query_vocab)
        self.keys_embedding = self.embedding_layer(self.keys_vocab)
        self.normal_embedding = self.embedding_layer(self.normal_vocab)
        # attention layer
        self.keys_mask = tf.expand_dims(tf.not_equal(keys_inputs, self.padding_value), axis=1)
        self.attention_out = self.attention_layer((self.query_embedding, self.keys_embedding, self.keys_mask))
        # mlp layer
        self.mlp_inputs = tf.concat([self.keys_embedding, self.query_embedding,
                                     self.attention_out, self.normal_embedding], axis=1)
        self.mlp_inputs = tf.reshape(self.mlp_inputs, shape=[-1, self.mlp_inputs.shape[1] * self.emb_size])
        self.mlp_out = self.mlp_layer(self.mlp_inputs)
        self.outputs = self.out_layer(self.mlp_out)
        return self.outputs

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "vocab_list": self.vocab_list,
                       "act_fun": self.act_fun, "reg_fun": self.reg_fun, "hidden_units": self.hidden_units,
                       "padding_value": self.padding_value})
        return config
