from recUtils.tf_utils import *


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, act_fun="relu", reg_fun=None, use_bn=False, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        # hyper parameters
        self.output_size = output_size
        self.act_fun = get_act_fun(act_fun, **kwargs)
        self.reg_fun = get_reg_fun(reg_fun, **kwargs)
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # initialize layers
        self.kernels = self.add_weight(name="weights",
                                       shape=[input_shape[-1], self.output_size],
                                       initializer=tf.keras.initializers.GlorotNormal(),
                                       regularizer=self.reg_fun,
                                       trainable=True)
        self.bias = self.add_weight(name="bias",
                                    shape=[self.output_size],
                                    initializer=tf.keras.initializers.GlorotNormal(),
                                    regularizer=self.reg_fun,
                                    trainable=True)
        if self.use_bn:
            self.bn_layers = tf.keras.layers.BatchNormalization()
        self.dropout_layers = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.activation_layers = tf.keras.layers.Activation(activation=self.act_fun)

    def call(self, inputs, **kwargs):
        fc = inputs @ self.kernels + self.bias
        if self.use_bn:
            fc = self.bn_layers(fc)
        fc = self.dropout_layers(fc)
        out = self.activation_layers(fc)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size, "act_fun": self.act_fun, "reg_fun": self.reg_fun,
                       "use_bn": self.use_bn, "dropout_rate": self.dropout_rate})
        return config
