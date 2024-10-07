import tensorflow as tf
from layers.vocab_layer import VocabLayer
from layers.fm_layer import FMLayer


class FM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.factor_dim = param_dict["factor_dim"]
        self.vocab_list = param_dict["vocab_list"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.fm_layer = FMLayer(param_dict["feature_size"], factor_dim=self.factor_dim)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        all_inputs = tf.concat(list(inputs.values()), axis=1)
        # vocab layer
        vocab = self.vocab_layer(all_inputs)
        # fm layer
        score = self.fm_layer(vocab)
        outputs = self.out_layer(score)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"factor_dim": self.factor_dim, "vocab_list": self.vocab_list})
        return config
