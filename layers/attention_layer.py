from layers.mlp_layer import *


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, param_dict, **kwargs):
        super(AttentionLayer, self).__init__()
        self.hidden_units = param_dict["hidden_units"]
        self.act_fun = param_dict["act_fun"]
        self.reg_fun = param_dict["reg_fun"]
        self.mask_value = -2 ** 32 + 1
        self.mlp_layer = MLPLayer(hidden_units=self.hidden_units, act_fun=self.act_fun, reg_fun=self.reg_fun)

    def call(self, inputs, **kwargs):
        self.query, self.keys, self.keys_mask = inputs
        self.queries = tf.tile(self.query, multiples=[1, tf.shape(self.keys)[1], 1])
        self.attention_mlp_inputs = tf.concat(
            [self.queries, self.keys, self.queries - self.keys, self.queries * self.keys],axis=-1)
        self.attention_mlp_outputs = self.mlp_layer(self.attention_mlp_inputs)
        self.attention_mlp_outputs = tf.reshape(self.attention_mlp_outputs , shape=[-1, 1, tf.shape(self.keys)[1]])
        self.attention_mlp_outputs = tf.where(self.keys_mask, self.attention_mlp_outputs, self.mask_value)
        self.attention_weights = tf.nn.softmax(self.attention_mlp_outputs, axis=2)
        self.attention_outputs = self.attention_weights @ self.keys
        return self.attention_outputs

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({"hidden_units": self.hidden_units, "act_fun": self.act_fun, "reg_fun": self.reg_fun})
        return config
