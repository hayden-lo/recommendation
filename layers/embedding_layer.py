from utils.tf_utils import *


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, emb_size=128, init_fun=tf.keras.initializers.RandomNormal(), reg_fun=None,
                 constraint=None, combiner="sum", **kwargs):
        super(EmbeddingLayer, self).__init__()
        self.feature_size = feature_size
        self.emb_size = emb_size
        self.initializer = get_init_fun(init_fun, **kwargs)
        self.regularizer = get_reg_fun(reg_fun, **kwargs)
        self.constraint = constraint
        self.combiner = combiner
        self.embedding_table = None

    def build(self, input_shape):
        self.embedding_table = self.add_weight(name="embedding",
                                               shape=[self.feature_size + 1, self.emb_size],
                                               initializer=self.initializer,
                                               regularizer=self.regularizer,
                                               constraint=self.constraint,
                                               trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "initializer": self.initializer,
                       "regularizer": self.regularizer, "constraint": self.constraint})
        return config
