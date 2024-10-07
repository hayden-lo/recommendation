from utils.tf_utils import *
from layers.dense_layer import DenseLayer


class FCLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, layer_name="fc", act_fun="relu", init_fun="random_normal", reg_fun=None,
                 use_bias=True, use_bn=False, dropout_rate=0.0, training=True, **kwargs):
        super(FCLayer, self).__init__()
        # hyper parameters
        self.output_size = output_size
        self.layer_name = layer_name
        self.act_fun = get_act_fun(act_fun, **kwargs)
        self.init_fun = get_init_fun(init_fun, **kwargs)
        self.reg_fun = get_reg_fun(reg_fun, **kwargs)
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.training = training
        self.dense_layer = None
        self.bn_layers = None
        self.dropout_layers = None
        self.activation_layers = None

    def build(self, input_shape):
        self.dense_layer = DenseLayer(output_size=self.output_size,
                                      init_fun=self.init_fun,
                                      reg_fun=self.reg_fun,
                                      use_bias=self.use_bias)
        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization(trainable=self.training)
        if self.dropout_rate > 0.0:
            self.dropout_layers = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(activation=self.act_fun)

    def call(self, inputs, **kwargs):
        fc_out = self.dense_layer(inputs)
        if self.use_bn:
            fc_out = self.bn_layers(fc_out)
        fc_out = self.activation_layers(fc_out)
        if self.dropout_rate > 0.0:
            fc_out = self.dropout_layers(fc_out)
        return fc_out
