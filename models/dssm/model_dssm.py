from layers.vocab_layer import *
from layers.embedding_layer import *
from layers.mlp_layer import *
from layers.similarity_layer import *


class DSSM(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.user_features = param_dict["user_features"]
        self.item_features = param_dict["item_features"]
        self.user_feats_size = param_dict["user_feats_size"]
        self.item_feats_size = param_dict["item_feats_size"]
        self.act_fun = param_dict["act_fun"]
        self.reg_fun = param_dict["reg_fun"]
        self.hidden_units = param_dict["hidden_units"]
        self.vocab_layer = VocabLayer(param_dict["vocab_list"])
        self.embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size)
        self.user_mlp_layer = MLPLayer(hidden_units=self.hidden_units, act_fun=self.act_fun, reg_fun=self.reg_fun)
        self.item_mlp_layer = MLPLayer(hidden_units=self.hidden_units, act_fun=self.act_fun, reg_fun=self.reg_fun)
        self.similarity_layer = CosineSimilarityLayer()
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        self.user_inputs = tf.concat([inputs[feat] for feat in self.user_features], axis=1)
        self.item_inputs = tf.concat([inputs[feat] for feat in self.item_features], axis=1)
        # vocab layer
        user_vocab = self.vocab_layer(self.user_inputs)
        item_vocab = self.vocab_layer(self.item_inputs)
        # user tower
        user_embeddings = self.embedding_layer(user_vocab)
        user_embeddings = tf.reshape(user_embeddings, shape=[-1, self.user_feats_size * self.emb_size])
        self.user_out = self.user_mlp_layer(user_embeddings)
        # item tower
        item_embeddings = self.embedding_layer(item_vocab)
        item_embeddings = tf.reshape(item_embeddings, shape=[-1, self.item_feats_size * self.emb_size])
        self.item_out = self.item_mlp_layer(item_embeddings)
        # cosine similarity
        sim_score = self.similarity_layer([self.user_out, self.item_out])
        # output layer
        self.outputs = self.out_layer(sim_score)
        return {"predictions": self.outputs, "user_embedding": self.user_out, "item_embedding": self.item_out}

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size,
                       "user_featuers": self.user_features, "item_features": self.item_features,
                       "act_fun": self.act_fun, "reg_fun": self.reg_fun, "user_feats_size": self.user_feats_size,
                       "item_feats_size": self.item_feats_size})
        return config
