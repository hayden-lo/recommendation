from rec_layers.layer_embedding import *


class LinearLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=1)

    def call(self, inputs, **kwargs):
        self.linear_weights = self.embedding_layer(inputs)
        self.linear_weights = tf.squeeze(self.linear_weights, axis=2)
        self.score = tf.reduce_sum(self.linear_weights, axis=1)
        return self.score

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size})
        return config
