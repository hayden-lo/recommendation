from layers.expert_layer import *
from utils.tf_utils import *


class MMoELayer(tf.keras.layers.Layer):
    def __init__(self, expert_num, target_num, expert_units, gate_units, layer_name="mmoe", expert_act_fun="relu",
                 gate_act_fun="softmax", expert_init_fun="random_normal", gate_init_fun="random_normal",
                 expert_reg_fun=None, gate_reg_fun=None, expert_use_bias=True, gate_use_bias=True, expert_use_bn=False,
                 gate_use_bn=False, expert_dropout_rate=0.0, gate_dropout_rate=0.0, training=True, **kwargs):
        super(MMoELayer, self).__init__()
        self.layer_name = layer_name
        self.expert_num = expert_num
        self.target_num = target_num
        self.expert_units = expert_units
        self.gate_units = gate_units
        self.expert_act_fun = expert_act_fun
        self.gate_act_fun = gate_act_fun
        self.expert_init_fun = expert_init_fun
        self.gate_init_fun = gate_init_fun
        self.expert_reg_fun = expert_reg_fun
        self.gate_reg_fun = gate_reg_fun
        self.expert_use_bias = expert_use_bias
        self.gate_use_bias = gate_use_bias
        self.expert_use_bn = expert_use_bn
        self.gate_use_bn = gate_use_bn
        self.expert_dropout_rate = expert_dropout_rate
        self.gate_dropout_rate = gate_dropout_rate
        self.training = training
        self.expert_layer = None
        self.gate_layer = None
        self.expert_outputs = None
        self.gate_outputs = None
        self.outputs = []

    def build(self, input_shape):
        self.expert_layer = ExpertLayer(expert_num=self.expert_num, expert_units=self.expert_units, layer_name="expert",
                                        act_fun=self.expert_act_fun, init_fun=self.expert_init_fun,
                                        reg_fun=self.expert_reg_fun, use_bias=self.expert_use_bias,
                                        use_bn=self.expert_use_bn, dropout_rate=self.expert_dropout_rate,
                                        training=self.training)
        self.gate_layer = ExpertLayer(expert_num=self.target_num, expert_units=self.gate_units, layer_name="gate",
                                      act_fun=self.gate_act_fun, init_fun=self.gate_init_fun,
                                      reg_fun=self.gate_reg_fun, use_bias=self.gate_use_bias,
                                      use_bn=self.gate_use_bn, dropout_rate=self.gate_dropout_rate,
                                      training=self.training)

    def call(self, inputs, *args, **kwargs):
        self.outputs = []
        # expert layer
        expert_outs_list = self.expert_layer(inputs)
        self.expert_outputs = tf.concat([tf.expand_dims(i, axis=-1) for i in expert_outs_list], axis=-1)
        # gate layer
        gate_outs_list = self.gate_layer(inputs)
        # mmoe output
        for gate in gate_outs_list:
            gate_weights = tf.tile(tf.expand_dims(gate, axis=1), multiples=[1, self.expert_units[-1], 1])
            weighted_expert_out = tf.reduce_sum(gate_weights * self.expert_outputs, axis=-1)
            self.outputs.append(weighted_expert_out)

        return self.outputs

    def get_config(self):
        config = super(MMoELayer, self).get_config()
        config.update({"layer_name": self.layer_name, "expert_num": self.expert_num, "target_num": self.target_num,
                       "expert_units": self.expert_units, "gate_units": self.gate_units,
                       "expert_act_fun": self.expert_act_fun, "gate_act_fun": self.gate_act_fun,
                       "expert_init_fun": self.expert_init_fun, "gate_init_fun": self.gate_init_fun,
                       "expert_reg_fun": self.expert_reg_fun, "gate_reg_fun": self.gate_reg_fun,
                       "expert_use_bias": self.expert_use_bias, "gate_use_bias": self.gate_use_bias,
                       "expert_use_bn": self.expert_use_bn, "gate_use_bn": self.gate_use_bn,
                       "expert_dropout_rate": self.expert_dropout_rate, "gate_dropout_rate": self.gate_dropout_rate,
                       "training": self.training})
        return config
