import os
import time
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from utils.log_utils import logger


def get_default_val(all_columns):
    default_val = []
    for i in all_columns.values():
        val = []
        if issubclass(i, str):
            val.append("")
        elif issubclass(i, float):
            val.append(0.0)
        elif issubclass(i, int):
            val.append(0)
        default_val.append(val)
    return default_val


def split_data(instance_data, sample_rate=0.7, delimiter=","):
    train_file = os.path.join(os.path.realpath(instance_data), "train.csv")
    test_file = os.path.join(os.path.realpath(instance_data), "test.csv")
    if os.path.exists(train_file) and os.path.exists(test_file):
        return train_file, test_file
    logger("Splitting Data")
    start = time.time()
    df = pd.read_csv(instance_data)
    train_df = df.sample(frac=sample_rate)
    test_df = df[~df.index.isin(train_df.index)]
    train_df.to_csv(train_file, header=False, index=False, sep=delimiter)
    test_df.to_csv(test_file, header=False, index=False, sep=delimiter)
    elapsed = time.time() - start
    logger(f"Splitting Data Elapse {round(elapsed / 60, 2)} minutes")
    return train_file, test_file


def get_valid_feats(param_dict):
    logger("Get valid features")
    start = time.time()
    df = pd.read_csv(param_dict["train_file"], names=param_dict["all_columns"].keys(), dtype=param_dict["all_columns"],
                     skiprows=1)
    valid_feat_list = []
    # target item
    valid_feat_list += ["click_seq_" + feat for feat, hit in df["movieId"].value_counts().items() if
                        hit >= param_dict["min_hits"]]
    # categorical features
    for cat_feat in param_dict["cat_columns"]:
        valid_feat_list += [cat_feat + "_" + feat for feat, hit in df[cat_feat].value_counts().items() if
                            hit >= param_dict["min_hits"]]
    # sequential features
    for seq_feat in param_dict["seq_columns"]:
        seq_dict = defaultdict(int)
        for record in df[seq_feat]:
            for element in str(record).split("|"):
                feat = seq_feat + "_" + element
                seq_dict[feat] += 1
        valid_feat_list += [k for k, v in seq_dict.items() if v >= param_dict["min_hits"]]
    elapsed = time.time() - start
    logger(f"Get Valid features elapse {round(elapsed / 60, 2)} minutes")
    return list(set(valid_feat_list))


def parse_data(row, param_dict):
    record = tf.io.decode_csv(row, record_defaults=get_default_val(param_dict["all_columns"]),
                              field_delim=param_dict["field_delimiter"])
    col2val = {col: val for col, val in zip(param_dict["all_columns"].keys(), record)}
    # target item
    tgt_dict = {"movieId": tf.expand_dims("click_seq_" + col2val["movieId"], axis=0)}
    # categorical features
    cat_dict = {cat: tf.expand_dims(cat + "_" + col2val[cat], axis=0) for cat in param_dict["cat_columns"]}
    # sequence features
    seq_dict = defaultdict(list)
    for seq, max_seq_num in param_dict["seq_columns"].items():
        seq_val = tf.strings.split([col2val[seq]], param_dict["seq_delimiter"]).values[:max_seq_num]
        seq_val = tf.strings.join([seq, seq_val], "_")
        seq_dict[seq] = tf.pad(seq_val, [[0, max_seq_num - tf.shape(seq_val)[0]]],
                               constant_values=param_dict["padding_value"])
    # feat_dict = {**tgt_dict, **cat_dict, **seq_dict}
    # user_inputs = tf.concat([feat_dict[feat] for feat in param_dict["user_features"]], axis=0)
    # item_inputs = tf.concat([feat_dict[feat] for feat in param_dict["item_features"]], axis=0)
    # features = {"user_inputs": user_inputs, "item_inputs": item_inputs}
    # features = tf.concat(list(feat_dict.values()), axis=0)
    features = {**tgt_dict, **cat_dict, **seq_dict}
    return features, col2val["label"]
