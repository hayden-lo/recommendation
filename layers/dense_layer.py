from utils.tf_utils import *


class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, output_size, init_fun="random_normal", reg_fun=None, use_bias=True, **kwargs):
        super(DenseLayer, self).__init__()
        # hyper parameters
        self.output_size = output_size
        self.init_fun = get_init_fun(init_fun, **kwargs)
        self.reg_fun = get_reg_fun(reg_fun, **kwargs)
        self.use_bias = use_bias
        self.kernels = None
        self.bias = None

    def build(self, input_shape):
        # initialize layers
        self.kernels = self.add_weight(name="weights",
                                       shape=[input_shape[-1], self.output_size],
                                       initializer=self.init_fun,
                                       regularizer=self.reg_fun,
                                       trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name="bias",
                                        shape=[self.output_size],
                                        initializer=self.init_fun,
                                        regularizer=self.reg_fun,
                                        trainable=True)

    def call(self, inputs, **kwargs):
        fc = inputs @ self.kernels
        if self.use_bias:
            fc = fc + self.bias
        return fc

    def get_config(self):
        config = super(DenseLayer, self).get_config()
        config.update({"output_size": self.output_size, "init_fun": self.init_fun, "reg_fun": self.reg_fun,
                       "use_bias": self.use_bias})
        return config
