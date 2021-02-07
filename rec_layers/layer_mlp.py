from rec_layers.layer_dense import *
from recUtils.tf_utils import *


class MLPLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, act_fun="relu", reg_fun=None, use_bn=False, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.act_fun = act_fun
        self.reg_fun = reg_fun
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.mlp_layer = tf.keras.Sequential()
        for i, unit in enumerate(self.hidden_units):
            if i == len(self.hidden_units) - 1:
                self.act_fun = None
            dense = DenseLayer(unit, act_fun=self.act_fun, reg_fun=self.reg_fun, use_bn=self.use_bn,
                               dropout_rate=self.dropout_rate)
            self.mlp_layer.add(dense)

    def call(self, inputs, **kwargs):
        return self.mlp_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({})
        return config
