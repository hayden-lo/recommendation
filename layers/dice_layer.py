import tensorflow as tf


class Dice(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-9):
        super(Dice, self).__init__()
        self.epsilon = epsilon
        self.alpha = None

    def build(self, input_shape):
        self.alpha = self.add_weight(name='dice_alpha',
                                     shape=input_shape,
                                     initializer=tf.initializers.truncated_normal(),
                                     trainable=True)

    def call(self, inputs, **kwargs):
        top = inputs - tf.reduce_mean(inputs, axis=0)
        bottom = tf.sqrt(tf.math.reduce_variance(inputs, axis=0) + self.epsilon)
        z = top / bottom
        # p = tf.sigmoid(z)
        p = tf.sigmoid(-z)
        return p * inputs + (1 - p) * self.apha * inputs
