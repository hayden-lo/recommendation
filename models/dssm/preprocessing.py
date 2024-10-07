import os
import pandas as pd
import tensorflow as tf
from collections import defaultdict


def split_data(param_dict):
    df = pd.read_csv(os.path.join(param_dict["data_dir"], param_dict["data_file"]))
    train_df = df.sample(frac=0.7)
    test_df = df[~df.index.isin(train_df.index)]
    train_file = os.path.join(param_dict["data_dir"], param_dict["train_file"])
    test_file = os.path.join(param_dict["data_dir"], param_dict["test_file"])
    train_df.to_csv(train_file, header=False, index=False, sep=param_dict["delimiter"])
    test_df.to_csv(test_file, header=False, index=False, sep=param_dict["delimiter"])
    return train_file, test_file


def get_valid_feats(param_dict):
    train_path = os.path.join(param_dict["data_dir"], param_dict["train_file"])
    df = pd.read_csv(train_path, names=param_dict["all_columns"],
                     dtype=dict(zip(param_dict["all_columns"], param_dict["data_type"])))
    valid_feat_list = []
    # target item
    valid_feat_list += ["click_seq_" + feat for feat, hit in df["movieId"].value_counts().items() if
                        hit >= param_dict["min_hit"]]
    # categorical features
    for cat_feat in param_dict["cat_columns"]:
        valid_feat_list += [cat_feat + "_" + feat for feat, hit in df[cat_feat].value_counts().items() if
                            hit >= param_dict["min_hit"]]
    # sequential features
    for seq_feat in param_dict["seq_columns"]:
        seq_dict = defaultdict(int)
        for record in df[seq_feat]:
            for element in str(record).split("|"):
                feat = seq_feat + "_" + element
                seq_dict[feat] += 1
        valid_feat_list += [k for k, v in seq_dict.items() if v >= param_dict["min_hit"]]
    return list(set(valid_feat_list))


def parse_data(row, param_dict):
    record = tf.io.decode_csv(row, record_defaults=param_dict["default_val"], field_delim=param_dict["delimiter"])
    col2val = {col: val for col, val in zip(param_dict["all_columns"], record)}
    # target item
    tgt_dict = {"movieId": tf.expand_dims("click_seq_" + col2val["movieId"], axis=0)}
    # categorical features
    cat_dict = {cat: tf.expand_dims(cat + "_" + col2val[cat], axis=0) for cat in param_dict["cat_columns"]}
    # sequence features
    seq_dict = defaultdict(list)
    for seq, max_seq_num in param_dict["seq_columns"].items():
        seq_val = tf.strings.split([col2val[seq]], "|").values[:max_seq_num]
        seq_val = tf.strings.join([seq, seq_val], "_")
        seq_dict[seq] = tf.pad(seq_val, [[0, max_seq_num - tf.shape(seq_val)[0]]],
                               constant_values=param_dict["padding_value"])
    feat_dict = {**tgt_dict, **cat_dict, **seq_dict}
    user_inputs = tf.concat([feat_dict[feat] for feat in param_dict["user_features"]], axis=0)
    item_inputs = tf.concat([feat_dict[feat] for feat in param_dict["item_features"]], axis=0)
    features = {"user_inputs": user_inputs, "item_inputs": item_inputs}
    return features, col2val["label"]
