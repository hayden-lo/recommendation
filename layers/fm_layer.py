from layers.embedding_layer import EmbeddingLayer
from utils.tf_utils import *


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, factor_dim=50, use_global_bias=True, init_fun=tf.keras.initializers.RandomNormal(),
                 reg_fun=None,
                 constraint=None, **kwargs):
        super(FMLayer, self).__init__()
        # hyper parameters
        self.feature_size = feature_size
        self.factor_dim = factor_dim
        self.use_global_bias = use_global_bias
        self.init_fun = get_init_fun(init_fun, **kwargs)
        self.reg_fun = get_reg_fun(reg_fun, **kwargs)
        self.constraint = constraint
        self.embedding_layer1 = EmbeddingLayer(feature_size=self.feature_size, emb_size=1, init_fun=self.init_fun,
                                               reg_fun=self.reg_fun, constraint=self.constraint)
        self.embedding_layer2 = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.factor_dim,
                                               init_fun=self.init_fun, reg_fun=self.reg_fun, constraint=self.constraint)
        self.global_bias = tf.keras.initializers.constant(0.0)([1])

    # def build(self, input_shape): self.embedding_layer1 = EmbeddingLayer(feature_size=input_shape, emb_size=1,
    # init_fun=self.init_fun, reg_fun=self.reg_fun, constraint=self.constraint) self.embedding_layer2 =
    # EmbeddingLayer(feature_size=input_shape, emb_size=self.factor_dim, init_fun=self.init_fun,
    # reg_fun=self.reg_fun, constraint=self.constraint) if self.use_global_bias: self.global_bias =
    # tf.keras.initializers.constant(0.0)([1])

    def call(self, inputs, **kwargs):
        # first order
        first_weights = self.embedding_layer1(inputs)
        first_weights = tf.squeeze(first_weights, axis=2)
        first_score = tf.reduce_sum(first_weights, axis=1)
        # second_order
        second_weights = self.embedding_layer2(inputs)
        sum_square = tf.square(tf.reduce_sum(second_weights, axis=1))
        square_sum = tf.reduce_sum(tf.square(second_weights), axis=1)
        second_score = 0.5 * tf.reduce_sum((sum_square - square_sum), axis=1)
        # fm score
        score = first_score + second_score
        if self.use_global_bias:
            score += self.use_global_bias
        return score
