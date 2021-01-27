from rec_layers.layer_embedding import *
from rec_layers.layer_dense import *
from rec_layers.layser_similarity import *
from rec_layers.layer_vocab import *
from rec_dssm.layer_input import *


class DSSM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.cat_columns = param_dict["cat_columns"]
        self.seq_columns = param_dict["seq_columns"]
        self.user_features = param_dict["user_features"]
        self.item_features = param_dict["item_features"]
        self.user_feats_size = param_dict["user_feats_size"]
        self.item_feats_size = param_dict["item_feats_size"]
        self.act_fun = param_dict["act_fun"]
        self.vocab_layer = VocabLayer(param_dict["vocab_list"])
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size)
        self.user_dense_layer1 = DenseLayer(256, self.act_fun)
        self.user_dense_layer2 = DenseLayer(128, self.act_fun)
        self.item_dense_layer1 = DenseLayer(256, self.act_fun)
        self.item_dense_layer2 = DenseLayer(128, self.act_fun)
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
        return self.outputs

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "act_fun": self.act_fun,
                       "reg_fun": self.reg_fun, "use_bn": self.use_bn, "dropout_rate": self.dropout_rate})
        return config


def get_DSSM(param_dict):
    vocab_layer = VocabLayer(param_dict["vocab_list"])
    embedding_layer = EmbeddingLayer(feature_size=param_dict["feature_size"], emb_size=param_dict["emb_size"])
    user_dense_layer1 = DenseLayer(256, param_dict["act_fun"])
    user_dense_layer2 = DenseLayer(128, param_dict["act_fun"])
    item_dense_layer1 = DenseLayer(256, param_dict["act_fun"])
    item_dense_layer2 = DenseLayer(128, param_dict["act_fun"])
    similarity_layer = CosineSimilarityLayer()
    out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)
    # define inputs
    user_inputs = tf.keras.Input(shape=[30], name="user_inputs", dtype=tf.string)
    item_inputs = tf.keras.Input(shape=[34], name="item_inputs", dtype=tf.string)
    user_vocab = vocab_layer(user_inputs)
    item_vocab = vocab_layer(item_inputs)
    # user tower
    user_embeddings = embedding_layer(user_vocab)
    user_embeddings = tf.reshape(user_embeddings, shape=[-1, param_dict["user_feats_size"] * param_dict["emb_size"]])
    user_out = user_dense_layer1(user_embeddings)
    user_out = user_dense_layer2(user_out)
    # item tower
    item_embeddings = embedding_layer(item_vocab)
    item_embeddings = tf.reshape(item_embeddings, shape=[-1, param_dict["item_feats_size"] * param_dict["emb_size"]])
    item_out = item_dense_layer1(item_embeddings)
    item_out = item_dense_layer2(item_out)
    # cosine similarity
    sim_score = similarity_layer([user_out, item_out])
    # output layer
    outputs = out_layer(sim_score)
    model = tf.keras.models.Model(inputs=[user_inputs, item_inputs], outputs=outputs)
    model.__setattr__("user_inputs", user_inputs)
    model.__setattr__("item_inputs", item_inputs)
    model.__setattr__("user_out", user_out)
    model.__setattr__("item_out", item_out)
    return model
