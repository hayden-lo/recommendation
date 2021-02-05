from rec_layers.layer_dense import *


class DIN(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.vocab_list = param_dict["vocab_list"]
        self.batch_size = param_dict["batch_size"]

    def call(self, inputs, training=None, mask=None):
        all_inputs = tf.concat(list(inputs.values()), axis=1)
        # vocab layer
        self.vocab = self.vocab_layer(all_inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "vocab_list": self.vocab_list,
                       "batch_size": self.batch_size})
        return config
