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
    if alias is None:
        return None
    if str.lower(alias) == "relu":
        return tf.nn.relu(**kwargs)
    if str.lower(alias) == "sigmoid":
        return tf.nn.sigmoid(**kwargs)
    if str.lower(alias) == "tanh":
        return tf.nn.tanh(**kwargs)
    if str.lower(alias) == "dice":
        alphas = tf.initializers.truncated_normal(**kwargs)
        bn_layer = tf.keras.layers.BatchNormalization(**kwargs)
        bn_out = bn_layer.apply(**kwargs)
        sig = tf.nn.sigmoid(bn_out)
        return alphas * (1.0 - sig) * pre_active + sig * pre_active


def get_reg_fun(alias, **kwargs):
    if alias is None:
        return None
    if not isinstance(alias, str):
        return alias
    if str.lower(alias) == "l1":
        return tf.keras.regularizers.L1(**kwargs)
    if str.lower(alias) == "l2":
        return tf.keras.regularizers.L2(**kwargs)


def get_init_fun(alias, **kwargs):
    if not isinstance(alias, str):
        return alias
    if alias == "glorot_normal":
        return tf.initializers.glorot_normal(**kwargs)
    if alias == "lecun_normal":
        return tf.initializers.lecun_normal(**kwargs)
    if alias == "lecun_uniform":
        return tf.initializers.lecun_uniform(**kwargs)
    if alias == "truncated_normal":
        return tf.initializers.truncated_normal(**kwargs)
    if alias == "random_normal" or alias == "normal":
        return tf.initializers.random_normal(**kwargs)
    if alias == "random_uniform" or alias == "uniform":
        return tf.initializers.random_uniform(**kwargs)


def get_early_stop():
    early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.0001, patience=1)
    return early_stop_callback


class TensorflowFix(tf.keras.callbacks.Callback):
    def __init__(self):
        super(TensorflowFix, self).__init__()
        self._supports_tf_logs = True
        self._backup_loss = None

    def on_train_begin(self, logs=None):
        self._backup_loss = {**self.model.loss}

    def on_train_batch_end(self, batch, logs=None):
        self.model.loss = self._backup_loss
