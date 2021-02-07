import tensorflow as tf


def get_optimizer(alias, learning_rate, **kwargs):
    if str.lower(alias) == "sgd":
        return tf.optimizers.SGD(learning_rate=learning_rate, **kwargs)
    if str.lower(alias) == "adam":
        return tf.optimizers.Adam(learning_rate=learning_rate, **kwargs)
    if str.lower(alias) == "nadam":
        return tf.optimizers.Nadam(learning_rate=learning_rate, **kwargs)


def get_loss_fun(alias, **kwargs):
    if str.lower(alias) == "binary_cross_entropy":
        return tf.losses.BinaryCrossentropy(**kwargs)
    if str.lower(alias) == "categorical_cross_entropy":
        return tf.losses.CategoricalCrossentropy(**kwargs)
    if str.lower(alias) == "mse":
        return tf.losses.MSE


def get_metrics(aliases, **kwargs):
    metrics_list = []
    for alias in aliases:
        if str.lower(alias) == "auc":
            metrics_list.append(tf.metrics.AUC(**kwargs))
        if str.lower(alias) == "binary_accuracy":
            metrics_list.append(tf.metrics.BinaryAccuracy(**kwargs))
        if str.lower(alias) == "categorical_accuracy":
            metrics_list.append(tf.metrics.CategoricalAccuracy(**kwargs))
        if str.lower(alias) == "top_k_categorical_accuracy":
            metrics_list.append(tf.metrics.TopKCategoricalAccuracy(**kwargs))
    return metrics_list


def get_act_fun(alias, **kwargs):
    if alias == None:
        return None
    if str.lower(alias) == "relu":
        return tf.nn.relu


def get_reg_fun(alias, **kwargs):
    if str.lower(alias) == "l1":
        return tf.keras.regularizers.L1(**kwargs)
    if str.lower(alias) == "l2":
        return tf.keras.regularizers.L2(**kwargs)


class TensorflowFix(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TensorflowFix, self).__init__()
        self._supports_tf_logs = True
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
