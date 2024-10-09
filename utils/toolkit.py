import datetime
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from collections import defaultdict
from utils.mysql_utils import MysqlClient


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


def get_valid_feats(param_dict):
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


def low_memory_df(df, low_level="min"):
    for col, d in zip(df.columns, df.dtypes):
        if np.issubdtype(d, np.integer):
            if low_level == "min":
                df[col] = df[col].astype(np.int16)
            if low_level == "median":
                df[col] = df[col].astype(np.int32)
        if np.issubdtype(d, np.floating):
            if low_level == "min":
                df[col] = df[col].astype(np.float16)
            if low_level == "median":
                df[col] = df[col].astype(np.float32)
    return df


def write_content(content_list, id_prefix, id_column, content_column, table):
    m = MysqlClient()
    exist_id_list = m.get_data(f"select {id_column} from {table}")[id_column].to_list()
    id_list = []

    for i in range(len(content_list)):
        content_id = ""
        while content_id == "" or content_id in exist_id_list:
            content_id = id_prefix + str(random.randint(0, 99999)).ljust(5, "0")
        id_list.append(content_id)

    insert_df = pd.DataFrame({id_column: id_list, content_column: content_list})
    year = datetime.now().year
    month = datetime.now().month
    day = datetime.now().day
    insert_df["create_date"] = datetime(year, month, day)
    m.insert_df(insert_df, table)


def get_duplicate():
    m = MysqlClient()
    df = m.get_data("select * from joke.dim_joke_di")
    content_list = []
    same_dict = defaultdict(set)
    for row in df.iterrows():
        content_list.append(row[1])

    for i in range(len(content_list)):
        for j in range(i + 1, len(content_list)):
            if content_list[i]["content"] == content_list[j]["content"]:
                same_dict[content_list[i]["content"]].add(content_list[i]["joke_id"])
                same_dict[content_list[i]["content"]].add(content_list[j]["joke_id"])

    return same_dict


def get_duplicate_keys(same_dict):
    duplicate_keys = []
    for v in same_dict.values():
        for i in range(1, len(v)):
            duplicate_keys.append(list(v)[i])
    return duplicate_keys


def round_up(n, d):
    tens = 10 ** d
    return round(n * tens) / tens
