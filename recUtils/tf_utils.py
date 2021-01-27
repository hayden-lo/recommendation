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
        return tf.losses.BinaryCrossentropy(from_logits=True, **kwargs)
    if str.lower(alias) == "categorical_cross_entropy":
        return tf.losses.CategoricalCrossentropy(from_logits=True, **kwargs)
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
    if str.lower(alias) == "relu":
        return tf.nn.relu
    tf.keras.layers.Activation()
