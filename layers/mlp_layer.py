from layers.fc_layer import *
from utils.tf_utils import *


class MLPLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_units, layer_name="mlp", act_fun="relu", init_fun="random_normal", reg_fun=None,
                 use_bias=True, use_bn=False, dropout_rate=0.0, training=True, **kwargs):
        super(MLPLayer, self).__init__()
        self.layer_name = layer_name
        self.hidden_units = hidden_units
        self.act_fun = act_fun
        self.init_fun = init_fun
        self.reg_fun = reg_fun
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.training = training
        self.mlp_layer = tf.keras.Sequential(name=self.layer_name)

    def build(self, input_shape):
        for i, unit in enumerate(self.hidden_units):
            act_fun = self.act_fun[i] if isinstance(self.act_fun, list) else self.act_fun
            init_fun = self.init_fun[i] if isinstance(self.init_fun, list) else self.init_fun
            reg_fun = self.reg_fun[i] if isinstance(self.reg_fun, list) else self.reg_fun
            use_bias = self.use_bias[i] if isinstance(self.use_bias, list) else self.use_bias
            use_bn = self.use_bn[i] if isinstance(self.use_bn, list) else self.use_bn
            dropout_rate = self.dropout_rate[i] if isinstance(self.dropout_rate, list) else self.dropout_rate
            fc_layer = FCLayer(unit, act_fun=act_fun, init_fun=init_fun, reg_fun=reg_fun, use_bias=use_bias,
                               use_bn=use_bn, dropout_rate=dropout_rate, training=self.training)
            self.mlp_layer.add(fc_layer)

    def call(self, inputs, **kwargs):
        return self.mlp_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"layer_name": self.layer_name, "hidden_units": self.hidden_units, "act_fun": self.act_fun,
                       "init_fun": self.init_fun, "reg_fun": self.reg_fun, "use_bias": self.use_bias,
                       "use_bn": self.use_bn, "dropout_rate": self.dropout_rate, "training": self.training})
        return config
