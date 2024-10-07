import tensorflow as tf


class VocabLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_list, use_hash=False, hash_size=50E8):
        super(VocabLayer, self).__init__()
        self.feat2id_table = None
        self.vocab_list = vocab_list
        self.use_hash = use_hash
        self.hash_size = hash_size

    def build(self, input_shape):
        id_list = tf.range(1, len(self.vocab_list) + 1)
        if self.use_hash:
            id_list = tf.strings.to_hash_bucket_strong(self.vocab_list, self.hash_size, [1, 2])
        feat2id_initializer = tf.lookup.KeyValueTensorInitializer(self.vocab_list, id_list)
        self.feat2id_table = tf.lookup.StaticHashTable(initializer=feat2id_initializer, default_value=0)

    def call(self, inputs, **kwargs):
        return self.feat2id_table.lookup(inputs)

    def get_config(self):
        config = super(VocabLayer, self).get_config()
        config.update({"vocab_list": self.vocab_list})
        return config
