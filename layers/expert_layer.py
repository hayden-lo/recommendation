from layers.mlp_layer import *
from utils.tf_utils import *


class ExpertLayer(tf.keras.layers.Layer):
    def __init__(self, expert_num, expert_units, layer_name="expert", act_fun="relu", init_fun="random_normal",
                 reg_fun=None, use_bias=True, use_bn=False, dropout_rate=0.0, training=True, **kwargs):
        super(ExpertLayer, self).__init__()
        self.layer_name = layer_name
        self.expert_num = expert_num
        self.expert_units = expert_units
        self.act_fun = act_fun
        self.init_fun = init_fun
        self.reg_fun = reg_fun
        self.use_bias = use_bias
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.training = training
        self.expert_outs_list = []
        self.expert_list = []

    def build(self, input_shape):
        for i, expert_num in enumerate(range(self.expert_num)):
            expert_block = MLPLayer(self.expert_units, layer_name=f"{self.layer_name}_{i}", act_fun=self.act_fun,
                                    init_fun=self.init_fun, reg_fun=self.reg_fun, use_bias=self.use_bias,
                                    use_bn=self.use_bn, dropout_rate=self.dropout_rate, training=self.training)
            self.expert_list.append(expert_block)

    def call(self, inputs, *args, **kwargs):
        self.expert_outs_list = []
        for expert_block in self.expert_list:
            expert_out = expert_block(inputs)
            self.expert_outs_list.append(expert_out)
        return self.expert_outs_list

    def get_config(self):
        config = super(ExpertLayer, self).get_config()
        config.update({"layer_name": self.layer_name, "expert_num": self.expert_num, "expert_units": self.expert_units,
                       "act_fun": self.act_fun, "init_fun": self.init_fun, "reg_fun": self.reg_fun,
                       "use_bias": self.use_bias, "use_bn": self.use_bn, "dropout_rate": self.dropout_rate,
                       "training": self.training})
        return config
