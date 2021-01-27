import tensorflow as tf


class VocabLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_list, **kwargs):
        super().__init__(**kwargs)
        self.vocab_list = vocab_list
        self.id_list = tf.range(1, len(self.vocab_list) + 1)

    def build(self, input_shape):
        self.feat2id_initializer = tf.lookup.KeyValueTensorInitializer(self.vocab_list, self.id_list)
        self.feat2id_table = tf.lookup.StaticHashTable(initializer=self.feat2id_initializer, default_value=0)

    def call(self, inputs, **kwargs):
        return self.feat2id_table.lookup(inputs)
