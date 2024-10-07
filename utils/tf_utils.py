import tensorflow as tf
from collections import defaultdict


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
    # if str.lower(alias) == "dice":
    #     alphas = tf.initializers.truncated_normal(**kwargs)
    #     bn_layer = tf.keras.layers.BatchNormalization(**kwargs)
    #     bn_out = bn_layer.apply(**kwargs)
    #     sig = tf.nn.sigmoid(bn_out)
    #     return alphas * (1.0 - sig) * pre_active + sig * pre_active


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


def get_vocab_dict(df, sparse_features, seq_features=None, seq_delimiter="|"):
    vocab_dict = defaultdict(int)
    # categorical features
    for feature in sparse_features:
        vocab_dict.update({f"{feature}_{val}": hit for val, hit in df[feature].value_counts().items()})
    # sequential features
    if seq_features is not None:
        for feature, max_len in seq_features.items():
            seq_dict = defaultdict(int)
            for val in df[feature]:
                for element in str(val).split(seq_delimiter)[:max_len]:
                    feat = feature + "_" + element
                    seq_dict[feat] += 1
            for k, v in seq_dict.items():
                vocab_dict[k] += v
    return vocab_dict


def get_csv_default_values(all_cols, sparse_features, seq_features=None, dense_features=None, mark_cols=None,
                           label_cols="label", weight_cols=None):
    default_records = []
    for col in all_cols:
        if col in sparse_features:
            default_records.append([""])
        elif seq_features is not None and col in seq_features:
            default_records.append([""])
        elif dense_features is not None and col in dense_features:
            default_records.append([0.0])
        elif mark_cols is not None and col in mark_cols:
            default_records.append([""])
        elif weight_cols is not None and col in weight_cols:
            default_records.append([0.0])
        elif col in label_cols:
            default_records.append([0.0])
        else:
            default_records.append([""])
    return default_records


def parse_csv_data(row, all_cols, sparse_features, seq_features=None, dense_features=None, field_delim="^",
                   seq_delimiter="|", padding_val="padding_value", mark_cols=None, label_cols="label",
                   weight_cols=None):
    # default record
    record_defaults = get_csv_default_values(all_cols, sparse_features, seq_features, dense_features, mark_cols,
                                             label_cols, weight_cols)
    # decode csv
    record = tf.io.decode_csv(row, record_defaults=record_defaults, field_delim=field_delim)
    col2val = {col: val for col, val in zip(all_cols, record)}
    # marks
    marks = {k: col2val[k] for k in mark_cols} if mark_cols is not None else {}
    # labels
    labels = [col2val[k] for k in label_cols] if isinstance(label_cols, list) else col2val[label_cols]
    # weights
    weights = [col2val[k] for k in weight_cols] if isinstance(weight_cols, list) else col2val[weight_cols]
    # categorical features
    sparse_dict = {}
    for feature in sparse_features:
        sparse_feat = tf.cast(col2val[feature], tf.string)
        val = feature + "_" + tf.where(tf.equal(sparse_feat, ""), padding_val, sparse_feat)
        sparse_dict[feature] = tf.expand_dims(val, axis=0)
    # dense features
    dense_dict = {}
    if dense_features is not None and len(dense_features) > 0:
        for feature in dense_features:
            dense_feat = tf.cast(col2val[feature], tf.float32)
            dense_dict[feature] = tf.expand_dims(dense_feat, axis=0)
    # sequence features
    seq_dict = defaultdict(list)
    if seq_features is not None and len(seq_features) > 0:
        for feature, max_seq_num in seq_features.items():
            seq_feat = tf.cast(col2val[feature], tf.string)
            seq_val = tf.strings.split([seq_feat], seq_delimiter).values[:max_seq_num]
            seq_val = tf.where(tf.equal(seq_val, ""), f"{feature}_{padding_val}", seq_val)
            seq_val = tf.pad(seq_val, [[0, max_seq_num - tf.shape(seq_val)[0]]],
                             constant_values=f"{feature}_{padding_val}")
            seq_val = tf.where(tf.equal(seq_val, f"{feature}_{padding_val}"), seq_val,
                               tf.strings.join([feature, seq_val], "_"))
            seq_dict[feature] = seq_val
    features = {**sparse_dict, **dense_dict, **seq_dict}
    return marks, features, tf.cast(labels, tf.float32), tf.cast(weights, tf.float32)
