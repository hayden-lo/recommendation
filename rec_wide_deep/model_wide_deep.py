from rec_layers.layer_vocab import *
from rec_layers.layer_embedding import *
from rec_layers.layer_dense import *


class WideDeep(tf.keras.models.Model):
    def __init__(self, param_dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_size = param_dict["feature_size"]
        self.emb_size = param_dict["emb_size"]
        self.input_size = param_dict["input_size"]
        self.vocab_list = param_dict["vocab_list"]
        self.batch_size = param_dict["batch_size"]
        self.act_fun = param_dict["act_fun"]
        self.reg_fun = param_dict["reg_fun"]
        self.sample_weight = param_dict["sample_weight"]
        self.from_logits = param_dict["from_logits"]
        self.vocab_layer = VocabLayer(self.vocab_list)
        self.wide_embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=1)
        self.deep_embedding_layer = EmbeddingLayer(feature_size=self.feature_size, emb_size=self.emb_size)
        self.deep_dense_layer1 = DenseLayer(64, self.act_fun, name="deep_dense1", reg_fun=self.reg_fun)
        self.deep_dense_layer2 = DenseLayer(32, self.act_fun, name="deep_dense2", reg_fun=self.reg_fun)
        self.deep_dense_layer3 = DenseLayer(1, self.act_fun, name="deep_dense3", reg_fun=self.reg_fun)
        self.out_layer = tf.keras.layers.Activation(activation=tf.keras.activations.sigmoid)

    def call(self, inputs, training=None, mask=None):
        tgt_inputs, cat_inputs, seq_inputs = list(inputs.values())
        all_inputs = tf.concat([tgt_inputs, cat_inputs, seq_inputs], axis=1)
        # vocab layer
        self.vocab = self.vocab_layer(all_inputs)
        # wide
        self.wide_weights = self.wide_embedding_layer(self.vocab)
        self.wide_score = tf.squeeze(self.wide_weights, axis=2)
        self.wide_score = tf.reduce_sum(self.wide_score, axis=1)
        # deep
        self.deep_embedding = self.deep_embedding_layer(self.vocab)
        self.deep_embedding = tf.reshape(self.deep_embedding, shape=[-1, self.input_size * self.emb_size])
        self.deep_output = self.deep_dense_layer1(self.deep_embedding)
        self.deep_output = self.deep_dense_layer2(self.deep_output)
        self.deep_score = self.deep_dense_layer3(self.deep_output)
        self.deep_score = tf.squeeze(self.deep_score)
        # combine score
        self.score = self.wide_score + self.deep_score
        self.outputs = self.out_layer(self.score)
        return self.outputs

    def train_step(self, data):
        x, y_true = data
        with tf.GradientTape() as tape:
            y_pred = self.call(inputs=x, training=True)
            bce_loss = tf.losses.binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)
            reweight_loss = -(self.sample_weight - 1) * y_true * tf.math.sigmoid(y_pred)
            loss = bce_loss + reweight_loss
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y_true, y_pred)
        loss_tracker = {"logloss": loss}
        metrics_tracker = {m.name: m.result() for m in self.metrics}
        return {**loss_tracker, **metrics_tracker}

    def get_config(self):
        config = super().get_config()
        config.update({"feature_size": self.feature_size, "emb_size": self.emb_size, "vocab_list": self.vocab_list,
                       "batch_size": self.batch_size, "act_fun": self.act_fun, "reg_fun": self.reg_fun})
        return config