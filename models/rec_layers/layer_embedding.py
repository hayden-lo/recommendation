import tensorflow as tf
from utils.tf_utils import *


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, emb_size=128, initializer=tf.keras.initializers.RandomNormal(), reg_fun=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.emb_size = emb_size
        self.initializer = initializer
        self.regularizer = get_reg_fun(reg_fun, **kwargs)

    def build(self, input_shape):
        self.embedding = self.add_weight(name="embedding",
                                         shape=[self.feature_size + 1, self.emb_size],
                                         initializer=self.initializer,
                                         regularizer=self.regularizer,
                                         trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "initializer": self.initializer})
        return config
