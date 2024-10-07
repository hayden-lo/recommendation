from layers.embedding_layer import *
from layers.mlp_layer import *
from layers.vocab_layer import *


class WideDeep(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.input_size = param_dict["input_size"]
        self.vocab_list = param_dict["vocab_list"]
        self.act_fun = param_dict["act_fun"]
        self.reg_fun = param_dict["reg_fun"]
        self.hidden_units = param_dict["hidden_units"]
        self.is_reweight = param_dict["is_reweight"]
        self.from_logits = param_dict["from_logits"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.wide_embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=1)
        self.deep_embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size)
        self.deep_mlp_layer = MLPLayer(hidden_units=self.hidden_units, act_fun=self.act_fun, reg_fun=self.reg_fun)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        all_inputs = tf.concat(list(inputs.values()), axis=1)
        # vocab layer
        self.vocab = self.vocab_layer(all_inputs)
        # wide
        self.wide_weights = self.wide_embedding_layer(self.vocab)
        self.wide_score = tf.squeeze(self.wide_weights, axis=2)
        self.wide_score = tf.reduce_sum(self.wide_score, axis=1)
        # deep
        self.deep_embedding = self.deep_embedding_layer(self.vocab)
        self.deep_embedding = tf.reshape(self.deep_embedding, shape=[-1, self.input_size * self.emb_size])
        self.deep_score = self.deep_mlp_layer(self.deep_embedding)
        self.deep_score = tf.squeeze(self.deep_score)
        # combine score
        self.score = self.wide_score + self.deep_score
        self.outputs = self.out_layer(self.score)
        return self.outputs

    def train_step(self, data):
        sample_weight = None
        if len(data) == 3 and self.is_reweight:
            x, y_true, sample_weight = data
        else:
            x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.call(inputs=x, training=True)
            bce_loss = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
            if self.is_reweight:
                bce_loss *= sample_weight
            loss = tf.reduce_mean(bce_loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)
        loss_tracker = {"logloss": loss}
        metrics_tracker = {m.name: m.result() for m in self.metrics}
        return {**loss_tracker, **metrics_tracker}

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "vocab_list": self.vocab_list,
                       "act_fun": self.act_fun, "reg_fun": self.reg_fun, "hidden_units": self.hidden_units,
                       "is_reweight":self.is_reweight,"from_logits":self.from_logits})
        return config
