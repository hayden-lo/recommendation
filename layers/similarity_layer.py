import tensorflow as tf


class CosineSimilarityLayer(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(CosineSimilarityLayer, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        query, documents = inputs
        query_norm = tf.norm(query, axis=self.axis)
        documents_norm = tf.norm(documents, axis=self.axis)
        cross = tf.reduce_sum((query * documents), axis=-1)
        out = tf.divide(cross, query_norm * documents_norm + 1e-8)
        return out

    def get_config(self):
        config = super(CosineSimilarityLayer, self).get_config()
        config.update({"axis": self.axis})
        return config