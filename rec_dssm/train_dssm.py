import numpy as np
from functools import partial
from rec_dssm.preprocessing import *
from rec_dssm.model_dssm import *
from recUtils.tf_utils import *


def run(param_dict):
    train_file, test_file = split_data(param_dict)
    valid_feats = get_valid_feats(param_dict)
    param_dict["vocab_list"] = valid_feats
    param_dict["feature_size"] = len(valid_feats)
    param_dict["user_feats_size"] = sum(
        [param_dict["max_seq_num"] if feat in param_dict["seq_columns"] else 1 for feat in param_dict["user_features"]])
    param_dict["item_feats_size"] = sum(
        [param_dict["max_seq_num"] if feat in param_dict["seq_columns"] else 1 for feat in param_dict["item_features"]])
    train_db = tf.data.TextLineDataset(train_file).map(partial(parse_data, param_dict=param_dict)).shuffle(
        param_dict["batch_size"] * 10).batch(param_dict["batch_size"])
    test_db = tf.data.TextLineDataset(test_file).map(partial(parse_data, param_dict=param_dict)).batch(
        param_dict["batch_size"])
    dssm_model = DSSM(param_dict)
    dssm_model.compile(optimizer=get_optimizer(param_dict["optimizer"], learning_rate=param_dict["learning_rate"]),
                       loss=get_loss_fun(param_dict["loss_fun"]),
                       metrics=get_metrics(param_dict["metrics"]))
    print("====================Model Training====================")
    dssm_model.fit(train_db, epochs=param_dict["epoch_num"])
    print("====================Model Evaluating====================")
    dssm_model.evaluate(test_db)
    # save model
    # dssm_model.save(filepath=param_dict["model_dir"],save_format="tf")
    # test
    print("====================Model Predicting====================")
    inputs = {"user_inputs": np.array([["12", "610"] + ["padding_value"] * 28]),
              "item_inputs": np.array([["34"] + ["padding_value"] * 33])}
    outputs = dssm_model.predict(inputs)
    for k, v in outputs:
        print("*" * 15 + k + "*" * 15)
        print(v)


if __name__ == '__main__':
    param_dict = {"model_dir": "model_dir",
                  "data_dir": "../recData/movieLens1m_201809",
                  "data_file": "data.csv",
                  "train_file": "train.csv",
                  "test_file": "test.csv",
                  "all_columns": ["userId", "movieId", "label", "screen_year", "rating_counts", "rating_mean",
                                  "genres", "click_seq"],
                  "cat_columns": ["screen_year", "rating_counts", "rating_mean"],
                  "seq_columns": ["genres", "click_seq"],
                  "user_features": ["click_seq"],
                  "item_features": ["movieId", "screen_year", "rating_counts", "rating_mean", "genres"],
                  "data_type": [np.str, np.str, np.float, np.str, np.str, np.str, np.str, np.str],
                  "default_val": [[""]] * 2 + [[0.0]] + [[""]] * 5,
                  "delimiter": ",",
                  "padding_value": "padding_value",
                  "min_hit": 3,
                  "act_fun": "relu",
                  "max_seq_num": 30,
                  "batch_size": 128,
                  "emb_size": 128,
                  "learning_rate": 0.01,
                  "optimizer": "adam",
                  "loss_fun": "binary_cross_entropy",
                  "metrics": ["auc"],
                  "epoch_num": 1}
    run(param_dict)
