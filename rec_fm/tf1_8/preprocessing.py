# Author: Hayden Lao
# Script Name: preprocessing
# Created Date: Oct 21st 2020
# Description: Preprocessing for FM model

import os
import pandas as pd
import tensorflow as tf
from functools import partial, reduce
from tensorflow.contrib.lookup import index_table_from_tensor


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
    # target_id
    valid_feat_list += ["click_seq" + "_" + feat for feat, hit in df.movieId.value_counts().items() if
                        hit >= param_dict["min_hit"]]
    # categorical features
    for cat_feat in param_dict["cat_columns"]:
        valid_feat_list += [cat_feat + "_" + str(feat) for feat, hit in df[cat_feat].value_counts().items() if
                            hit >= param_dict["min_hit"]]
    # sequential features
    for seq_feat in param_dict["seq_columns"]:
        seq_dict = {}
        for record in df[seq_feat]:
            for element in str(record).split("|"):
                feat = seq_feat + "_" + element
                seq_dict[feat] = 1 if feat not in seq_dict else seq_dict[feat] + 1
        valid_feat_list += [k for k, v in seq_dict.items() if v >= param_dict["min_hit"]]
    return list(set(valid_feat_list))


def parse_csv(row, param_dict):
    record = tf.decode_csv(row, record_defaults=param_dict["default_val"], field_delim=param_dict["delimiter"])
    col2val = {col: val for col, val in zip(param_dict["all_columns"], record)}
    # target item
    target_id = param_dict["feat_id_table"].lookup("click_seq_" + col2val["movieId"])
    # categorical features
    cat_list = [param_dict["feat_id_table"].lookup(cat + "_" + col2val[cat]) for cat in param_dict["cat_columns"]]
    cat_feats = tf.stack(cat_list, axis=0)
    # sequence features
    seq_list = []
    for seq in param_dict["seq_columns"]:
        seq_feat = tf.string_split([col2val[seq]], "|").values[:param_dict["max_seq_num"]]
        seq_feat = tf.string_join([seq, seq_feat], "_")
        seq_feat_id = param_dict["feat_id_table"].lookup(seq_feat)
        seq_val = tf.pad(seq_feat_id, [[0, param_dict["max_seq_num"] - tf.shape(seq_feat_id)[0]]])
        seq_list.append(seq_val)
    seq_feats = tf.concat(seq_list, axis=0)
    # seq_feat = tf.string_split([col2val["click_seq"]], "|").values[:param_dict["max_seq_num"]]
    # seq_feat = tf.string_join(["click_seq", seq_feat], "_")
    # seq_feat_id = param_dict["feat_id_table"].lookup(seq_feat)
    # seq_feats = tf.pad(seq_feat_id, [[0, param_dict["max_seq_num"] - tf.shape(seq_feat_id)[0]]])
    features = {"target_id": target_id,
                "cat_feats": cat_feats,
                "seq_feats": seq_feats}
    return features, col2val["label"]


def input_fn(data_file, param_dict):
    dataset = tf.data.TextLineDataset(data_file)
    valid_feat_list = get_valid_feats(param_dict)
    feat_id_table = index_table_from_tensor(mapping=valid_feat_list, default_value=0)
    param_dict["feat_id_table"] = feat_id_table
    dataset = dataset.map(partial(parse_csv, param_dict=param_dict)).batch(param_dict["batch_size"])
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    features, labels = iterator.get_next()
    return features, labels
