import numpy as np
from functools import partial
from rec_fm.preprocessing import *
from rec_fm.model_fm import *
from recUtils.tf_utils import *


def run(param_dict):
    train_file, test_file = split_data(param_dict)
    valid_feats = get_valid_feats(param_dict)
    param_dict["vocab_list"] = valid_feats
    param_dict["feature_size"] = len(valid_feats)
    train_db = tf.data.TextLineDataset(train_file).map(partial(parse_data, param_dict=param_dict)).shuffle(
        param_dict["batch_size"] * 10).batch(param_dict["batch_size"])
    test_db = tf.data.TextLineDataset(test_file).map(partial(parse_data, param_dict=param_dict)).batch(
        param_dict["batch_size"])
    fm_model = FM(param_dict)
    fm_model.compile(optimizer=get_optimizer(param_dict["optimizer"], learning_rate=param_dict["learning_rate"]),
                     loss=get_loss_fun(param_dict["loss_fun"], from_logits=param_dict["from_logits"]),
                     metrics=get_metrics(param_dict["metrics"]))
    print("====================Model Training====================")
    fm_model.fit(train_db, epochs=param_dict["epoch_num"])
    print("====================Model Evaluating====================")
    # fm_model.evaluate(test_db)
    # save model
    fm_model.save(filepath=param_dict["model_dir"], overwrite=True, save_format="tf")


if __name__ == "__main__":
    param_dict = {"model_dir": "./model_dir",
                  "data_dir": "../recData/movieLens1m_201809",
                  "data_file": "data.csv",
                  "train_file": "train.csv",
                  "test_file": "test.csv",
                  "all_columns": ["userId", "movieId", "label", "screen_year", "rating_counts", "rating_mean",
                                  "genres", "click_seq"],
                  "cat_columns": ["screen_year", "rating_counts", "rating_mean"],
                  "seq_columns": {"genres": 5, "click_seq": 30},
                  "data_type": [np.str, np.str, np.float, np.str, np.str, np.str, np.str, np.str],
                  "default_val": [[""]] * 2 + [[0.0]] + [[""]] * 5,
                  "delimiter": ",",
                  "padding_value": "padding_value",
                  "min_hit": 3,
                  "reg_fun": "l2",
                  "batch_size": 128,
                  "factor_dim": 50,
                  "learning_rate": 0.001,
                  "optimizer": "adam",
                  "loss_fun": "binary_cross_entropy",
                  "from_logits": False,
                  "metrics": ["auc"],
                  "epoch_num": 1}
    run(param_dict)
    print("====================Model Predicting====================")
    fm_model = tf.keras.models.load_model(param_dict["model_dir"])
    inputs = {"tgt_inputs": np.array([["movieId_157"]]),
              "cat_inputs": np.array([["screen_year_7", "rating_counts_4", "rating_mean_2"]]),
              "seq_inputs": np.array([["click_seq_2492", "click_seq_2012", "click_seq_2478", "click_seq_553",
                                       "click_seq_157", "click_seq_3053", "click_seq_1298", "click_seq_3448",
                                       "click_seq_151", "click_seq_1090", "click_seq_1224", "click_seq_5060",
                                       "click_seq_527", "click_seq_3147", "click_seq_2353", "click_seq_47",
                                       "click_seq_593", "click_seq_3033", "click_seq_1206", "click_seq_3702",
                                       "click_seq_1240", "click_seq_1270", "click_seq_2291", "click_seq_163",
                                       "click_seq_1226", "click_seq_943", "click_seq_1265", "click_seq_3273",
                                       "click_seq_1625", "click_seq_1092", "genres_Comedy", "genres_War"] +
                                      ["padding_value"] * 3])}
    outputs = fm_model.predict(inputs)
    print(outputs)