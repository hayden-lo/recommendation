from rec_layers.layer_embedding import *
from rec_layers.layer_dense import *
from rec_layers.layser_similarity import *
from rec_layers.layer_vocab import *


class DSSM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.user_feats_size = param_dict["user_feats_size"]
        self.item_feats_size = param_dict["item_feats_size"]
        self.act_fun = param_dict["act_fun"]
        self.vocab_layer = VocabLayer(param_dict["vocab_list"])
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size)
        self.user_dense_layer1 = DenseLayer(256, self.act_fun, name="user_dense1")
        self.user_dense_layer2 = DenseLayer(128, self.act_fun, name="user_dense2")
        self.item_dense_layer1 = DenseLayer(256, self.act_fun, name="item_dense1")
        self.item_dense_layer2 = DenseLayer(128, self.act_fun, name="item_dense2")
        self.similarity_layer = CosineSimilarityLayer()
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        self.user_inputs = inputs["user_inputs"]
        self.item_inputs = inputs["item_inputs"]
        user_vocab = self.vocab_layer(self.user_inputs)
        item_vocab = self.vocab_layer(self.item_inputs)
        # user tower
        user_embeddings = self.embedding_layer(user_vocab)
        user_embeddings = tf.reshape(user_embeddings, shape=[-1, self.user_feats_size * self.emb_size])
        user_out = self.user_dense_layer1(user_embeddings)
        self.user_out = self.user_dense_layer2(user_out)
        # item tower
        item_embeddings = self.embedding_layer(item_vocab)
        item_embeddings = tf.reshape(item_embeddings, shape=[-1, self.item_feats_size * self.emb_size])
        item_out = self.item_dense_layer1(item_embeddings)
        self.item_out = self.item_dense_layer2(item_out)
        # cosine similarity
        sim_score = self.similarity_layer([user_out, item_out])
        # output layer
        self.outputs = self.out_layer(sim_score)
        return {"preditions": self.outputs, "user_embedding": self.user_out, "item_embedding": self.item_out}

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "act_fun": self.act_fun,
                       "reg_fun": self.reg_fun, "use_bn": self.use_bn, "dropout_rate": self.dropout_rate})
        return config