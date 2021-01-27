import tensorflow as tf


class InputLayer(tf.keras.layers.Layer):
    def __init__(self, user_features, item_features, **kwargs):
        super().__init__(**kwargs)
        self.user_features = user_features
        self.item_features = item_features

    def call(self, inputs, **kwargs):
        user_inputs = tf.concat(
            [tf.expand_dims(tf.keras.Input(shape=[None], name=feat, dtype=tf.string), axis=1) for feat in
             self.user_features], axis=1)
        item_inputs = tf.concat(
            [tf.expand_dims(tf.keras.Input(shape=[None], name=feat, dtype=tf.string), axis=1) for feat in
             self.item_features], axis=1)
        return user_inputs, item_inputs
