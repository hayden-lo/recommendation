import numpy as np
from functools import partial
from utils.bak.preprocessing import *
from rec_dssm.model_dssm import *
from utils.tf_utils import *


def run(param_dict):
    train_file, test_file = split_data(param_dict)
    valid_feats = get_valid_feats(param_dict)
    param_dict["vocab_list"] = valid_feats
    param_dict["feature_size"] = len(valid_feats)
    param_dict["user_feats_size"] = sum(
        [param_dict["seq_columns"][feat] if feat in param_dict["seq_columns"] else 1 for feat in
         param_dict["user_features"]])
    param_dict["item_feats_size"] = sum(
        [param_dict["seq_columns"][feat] if feat in param_dict["seq_columns"] else 1 for feat in
         param_dict["item_features"]])
    train_db = tf.data.TextLineDataset(train_file).map(partial(parse_data, param_dict=param_dict)).shuffle(
        param_dict["batch_size"] * 10).batch(param_dict["batch_size"])
    test_db = tf.data.TextLineDataset(test_file).map(partial(parse_data, param_dict=param_dict)).batch(
        param_dict["batch_size"])
    dssm_model = DSSM(param_dict)
    dssm_model.compile(optimizer=get_optimizer(param_dict["optimizer"], learning_rate=param_dict["learning_rate"]),
                       loss={
                           "predictions": get_loss_fun(param_dict["loss_fun"], from_logits=param_dict["from_logits"])},
                       metrics={"predictions": get_metrics(param_dict["metrics"])})
    print("====================Model Training====================")
    dssm_model.fit(train_db, epochs=param_dict["epoch_num"], callbacks=[TensorflowFix()])
    print("====================Model Evaluating====================")
    dssm_model.evaluate(test_db)
    # save model
    dssm_model.save(filepath=param_dict["model_dir"], overwrite=True, save_format="tf")


if __name__ == '__main__':
    param_dict = {"model_dir": "./model_dir",
                  "data_dir": "../data/movieLens1m_201809",
                  "data_file": "data.csv",
                  "train_file": "train.csv",
                  "test_file": "test.csv",
                  "all_columns": ["userId", "movieId", "label", "rating", "screen_year", "rating_counts", "rating_mean",
                                  "genres", "click_seq"],
                  "cat_columns": ["screen_year", "rating_counts", "rating_mean"],
                  "seq_columns": {"genres": 5, "click_seq": 30},
                  "user_features": ["click_seq"],
                  "item_features": ["movieId", "screen_year", "rating_counts", "rating_mean", "genres"],
                  "data_type": [np.str, np.str, np.float, np.float, np.str, np.str, np.str, np.str, np.str],
                  "default_val": [[""]] * 2 + [[0.0]] * 2 + [[""]] * 5,
                  "delimiter": ",",
                  "padding_value": "padding_value",
                  "min_hit": 3,
                  "act_fun": "relu",
                  "reg_fun": "l2",
                  "hidden_units": [256, 128],
                  "batch_size": 128,
                  "emb_size": 128,
                  "learning_rate": 0.01,
                  "optimizer": "adam",
                  "loss_fun": "binary_cross_entropy",
                  "from_logits": False,
                  "metrics": ["auc"],
                  "epoch_num": 1}
    run(param_dict)
    print("====================Model Predicting====================")
    dssm_model = tf.keras.models.load_model(param_dict["model_dir"])
    inputs = {"movieId": np.array([["click_seq_157"]]),
              "screen_year": np.array([["screen_year_7"]]),
              "rating_counts": np.array([["rating_counts_4"]]),
              "rating_mean": np.array([["rating_mean_2"]]),
              "click_seq": np.array([["click_seq_2492", "click_seq_2012", "click_seq_2478", "click_seq_553",
                                      "click_seq_157", "click_seq_3053", "click_seq_1298", "click_seq_3448",
                                      "click_seq_151", "click_seq_1090", "click_seq_1224", "click_seq_5060",
                                      "click_seq_527", "click_seq_3147", "click_seq_2353", "click_seq_47",
                                      "click_seq_593", "click_seq_3033", "click_seq_1206", "click_seq_3702",
                                      "click_seq_1240", "click_seq_1270", "click_seq_2291", "click_seq_163",
                                      "click_seq_1226", "click_seq_943", "click_seq_1265", "click_seq_3273",
                                      "click_seq_1625", "click_seq_1092"]]),
              "genres": np.array([["genres_Comedy", "genres_War"] + ["padding_value"] * 3])}
    outputs = dssm_model.predict(inputs)
    print(outputs["predictions"])
    print("====================User Embedding====================")
    print(outputs["user_embedding"])
    print("====================Item Embedding====================")
    print(outputs["item_embedding"])
