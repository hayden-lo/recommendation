from functools import partial
from env.configuration import debug_inputs
from models.fm.config_fm import params
from models.fm.model_fm import *
from utils.toolkit import *
from utils.tf_utils import *

pd.options.display.max_columns = 100
pd.options.display.width = 500


def train(param_dict):
    param_dict["vocab_list"] = get_valid_feats(param_dict)
    param_dict["feature_size"] = len(param_dict["vocab_list"])
    param_dict["factor_dim"] = param_dict["factor_dim"]
    train_db = tf.data.TextLineDataset(param_dict["train_file"]).skip(1).map(
        partial(parse_data, param_dict=param_dict)).shuffle(param_dict["batch_size"] * 10).batch(
        param_dict["batch_size"])
    test_db = tf.data.TextLineDataset(param_dict["test_file"]).skip(1).map(
        partial(parse_data, param_dict=param_dict)).batch(param_dict["batch_size"])
    fm_model = FM(param_dict)
    fm_model.compile(optimizer=get_optimizer(param_dict["optimizer"], learning_rate=param_dict["learning_rate"]),
                     loss=get_loss_fun(param_dict["loss_function"], from_logits=param_dict["from_logits"]),
                     metrics=get_metrics(param_dict["metrics"]))
    logger("Model training")
    train_start = time.time()
    fm_model.fit(train_db, epochs=param_dict["epoch_num"], validation_data=test_db, callbacks=param_dict["callbacks"])
    logger(f"Train elapse {round((time.time() - train_start) / 60, 2)} minutes")
    # save model
    fm_model.save(filepath=param_dict["model_dir"], overwrite=True, save_format="tf")


if __name__ == "__main__":
    if params["mode"] == "train":
        train(params)
    if params["mode"] == "debug":
        load_model_start = time.time()
        logger("Model loading")
        model = tf.keras.models.load_model(params["model_dir"])
        logger(f"Load model elapsed {round((time.time() - load_model_start) / 60, 2)}")
        outputs = model.predict(debug_inputs)
        print(outputs)
    # if params["mode"] == "predict":
    #     fm_model = tf.keras.models.load_model(param_dict["model_dir"])
    #     print("====================Constructing Inputs====================")
    #     start = time.time()
    #     inputs = get_inputs(user_id=param_dict["user_id"], param_dict=params)
    #     elapsed = round(time.time() - start, 2)
    #     print("====================Constructing Inputs Elaspe {} seconds====================".format(elapsed))
    #     outputs = fm_model.predict(inputs)
    #     recom_df = get_recommendations(inputs=inputs, outputs=outputs, param_dict=param_dict)
    #     filter1 = (~recom_df["genres"].str.contains("Documentary"))
    #     filter2 = (recom_df["screen_year"] >= 1990)
    #     filter3 = (recom_df["genres"] != "(no genres listed)")
    #     filter4 = (recom_df["rating_counts"] > 3000)
    #     recom_df = recom_df[filter2 & filter3 & filter4]
    #     print(recom_df.head(20))
