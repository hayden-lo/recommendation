import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, feature_size, emb_size=128, initializer=tf.keras.initializers.RandomNormal, **kwargs):
        super().__init__(**kwargs)
        self.feature_size = feature_size
        self.emb_size = emb_size
        self.initializer = initializer

    def build(self, input_shape):
        self.embedding = self.add_weight(shape=[self.feature_size + 1, self.emb_size],
                                         initializer=self.initializer,
                                         trainable=True)

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.embedding, inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.embedding, "initializer": self.initializer})
        return config
