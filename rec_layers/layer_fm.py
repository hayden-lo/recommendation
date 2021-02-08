from rec_layers.layer_embedding import *


class FMLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, factor_dim=50, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.factor_dim = factor_dim
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.factor_dim)

    def call(self, inputs, **kwargs):
        self.fm_weights = self.embedding_layer(inputs)
        self.sum_square = tf.square(tf.reduce_sum(self.fm_weights, axis=1))
        self.square_sum = tf.reduce_sum(tf.square(self.fm_weights), axis=1)
        self.score = 0.5 * tf.reduce_sum((self.sum_square - self.square_sum), axis=1)
        return self.score

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "factor_dim": self.factor_dim})
        return config
