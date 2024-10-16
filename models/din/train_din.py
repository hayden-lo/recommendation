from functools import partial
from config.configs import *
from models.din.model_din import DIN
from utils.bak.get_inputs import get_inputs
from utils.bak.predict_methods import get_recommendations
from utils.bak.preprocessing import *
from utils.tf_utils import *


def run(param_dict):
    train_file, test_file = split_data(param_dict)
    valid_feats = get_valid_feats(param_dict)
    param_dict["vocab_list"] = valid_feats
    param_dict["feature_size"] = len(valid_feats)
    train_db = tf.data.TextLineDataset(train_file).map(partial(parse_data, param_dict=param_dict)).shuffle(
        param_dict["batch_size"] * 10).batch(param_dict["batch_size"])
    test_db = tf.data.TextLineDataset(test_file).map(partial(parse_data, param_dict=param_dict)).batch(
        param_dict["batch_size"])
    din_model = DIN(param_dict)
    din_model.compile(optimizer=get_optimizer(param_dict["optimizer"], learning_rate=param_dict["learning_rate"]),
                      loss=get_loss_fun(param_dict["loss_fun"], from_logits=param_dict["from_logits"]),
                      metrics=get_metrics(param_dict["metrics"]))
    print("====================Model Training====================")
    din_model.fit(train_db, epochs=param_dict["epoch_num"])
    print("====================Model Evaluating====================")
    din_model.evaluate(test_db)
    # save model
    din_model.save(filepath=param_dict["model_dir"], overwrite=True, save_format="tf")


if __name__ == "__main__":
    specific_dict = {"mode": "dev", "hidden_units": [64, 32, 1], "epoch_num": 5}
    train_dict = {"min_hit": 3000, "batch_size": 512, "learning_rate": 0.001, "epoch_num": 5,
                  "callbacks": [get_early_stop()]}
    predict_params["user_id"] = 999998
    param_dict = {**universal_params, **specific_dict}
    if param_dict["mode"] == "train":
        param_dict = {**param_dict, **train_params, **train_dict}
    if param_dict["mode"] in ("train", "dev"):
        run(param_dict)
        print("====================Model Predicting====================")
        din_model = tf.keras.models.load_model(param_dict["model_dir"])
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
                  "genres": np.array([["genres_Comedy", "genres_War"] + ["padding_value"] * 18])}
        outputs = din_model.predict(inputs)
        print(outputs)
    if param_dict["mode"] == "predict":
        param_dict = {**param_dict, **predict_params}
        din_model = tf.keras.models.load_model(param_dict["model_dir"])
        print("====================Constructing Inputs====================")
        start = time.time()
        inputs = get_inputs(user_id=param_dict["user_id"], param_dict=param_dict)
        elasped = round(time.time() - start, 2)
        print("====================Constructing Inputs Elaspe {} seconds====================".format(elasped))
        outputs = din_model.predict(inputs)
        recom_df = get_recommendations(inputs=inputs, outputs=outputs, param_dict=param_dict)
        # filter1 = (~recom_df["genres"].str.contains("Documentary"))
        # filter2 = (recom_df["screen_year"] >= 2000)
        # filter3 = (recom_df["genres"] != "(no genres listed)")
        # recom_df = recom_df[filter1 & filter2 & filter3]
        print(recom_df.head(20))
