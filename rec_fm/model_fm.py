from rec_layers.layer_fm import *
from rec_layers.layer_linear import *
from rec_layers.layer_vocab import *


class FM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.factor_dim = param_dict["factor_dim"]
        self.vocab_list = param_dict["vocab_list"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.linear_layer = LinearLayer(self.feature_size)
        self.fm_layer = FMLayer(self.feature_size, factor_dim=self.factor_dim)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        all_inputs = tf.concat(list(inputs.values()), axis=1)
        # vocab layer
        self.vocab = self.vocab_layer(all_inputs)
        # global bias
        self.global_bias = tf.keras.initializers.constant(0.0)([1])
        # first order
        self.first_score = self.linear_layer(self.vocab)
        # second_order
        self.second_score = self.fm_layer(self.vocab)
        # fm score
        self.score = self.global_bias + self.first_score + self.second_score
        self.outputs = self.out_layer(self.score)
        return self.outputs

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "factor_dim": self.factor_dim,
                       "vocab_list": self.vocab_list, })
        return config
