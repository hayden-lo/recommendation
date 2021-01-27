# Author: Hayden Lao
# Script Name: train_fm
# Created Date: Oct 24th 2020
# Description: Main entrance for FM model

import time
import shutil
import argparse
import numpy as np
from rec_fm.tf1_8.preprocessing import *
from rec_fm.tf1_8.model_fm import FM

# set parameters
parser = argparse.ArgumentParser(description="rec-fm")
parser.add_argument("--model_dir", type=str, default=r"D:\git\github\recommendation\rec_fm\model_dir",
                    help="model directory")
parser.add_argument("--data_dir", type=str, default=r"D:\git\github\recommendation\recData\movieLens1m_201809",
                    help="data directory")
parser.add_argument("--train_file", type=str, default=r"train_data.csv", help="train file")
parser.add_argument("--test_file", type=str, default=r"test_data.csv", help="test file")
parser.add_argument("--learning_rate", type=float, default=0.001, help=r"learning rate")
parser.add_argument("--batch_size", type=int, default=128, help=r"batch size")
parser.add_argument("--max_seq_num", type=int, default=30, help=r"maximum sequence length")
parser.add_argument("--normal_mean", type=float, default=0.0, help=r"mean for initializing weight")
parser.add_argument("--normal_dev", type=float, default=1.0, help=r"deviation for initializing weight")
parser.add_argument("--factor_dim", type=int, default=50, help=r"factorization dimension")
parser.add_argument("--l2_reg", type=float, default=0.01, help=r"l2 regularization parameter")
parser.add_argument("--epoch_num", type=int, default=3, help=r"epoch number")
parser.add_argument("--cat_size", type=int, default=3, help=r"categorical feature number")
namespace, args = parser.parse_known_args()


def model(param_dict):
    run_config = None
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=param_dict["model_dir"],
        params=param_dict,
        config=run_config
    )
    return estimator


def model_fn(features, labels, mode, params):
    fm_model = FM(features, param_dict=params)
    predictions = {"predictions": fm_model.predictions}
    eval_metrics = {"auc": tf.metrics.auc(labels, fm_model.predictions),
                    "recall": tf.metrics.recall(labels, fm_model.predictions)}
    loss = fm_model.get_loss(labels)
    train_op = fm_model.get_train_op(loss)
    estimator_entity = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op,
                                                  eval_metric_ops=eval_metrics)
    return estimator_entity


def run(_):
    tf.logging.info("*" * 15 + "Parameters:{}".format(param_dict) + "*" * 15)
    train_file, test_file = split_data(param_dict)
    valid_feats = get_valid_feats(param_dict)
    param_dict["feature_size"] = len(valid_feats)
    param_dict["cat_size"] = len(param_dict["cat_columns"])
    param_dict["seq_size"] = len(param_dict["seq_columns"])
    train_input = lambda: input_fn(train_file, param_dict)
    test_input = lambda: input_fn(test_file, param_dict)
    estimator = model(param_dict)
    for i in range(param_dict["epoch_num"]):
        tf.logging.info("=" * 15 + "Epoch {}".format(i + 1) + "=" * 15)
        epoch_start = time.time()
        estimator.train(input_fn=train_input)
        eval_start = time.time()
        metrics = estimator.evaluate(input_fn=test_input)
        eval_elaspe = round((time.time() - eval_start) / 60, 2)
        tf.logging.info("*" * 15 + "Evaluation elapsed: {} mins".format(eval_elaspe) + "*" * 15)
        tf.logging.info("*" * 15 + "Evaluation result" + "*" * 15)
        tf.logging.info(metrics)
        epoch_elapse = round((time.time() - epoch_start) / 60, 2)
        tf.logging.info("*" * 15 + "Epoch elapsed: {} mins".format(epoch_elapse) + "*" * 15)


if __name__ == '__main__':
    param_dict = {"model_dir": "model_dir",
                  "data_dir": "../../recData/movieLens1m_201809",
                  "data_file": "data.csv",
                  "train_file": "train.csv",
                  "test_file": "test.csv",
                  "all_columns": ["userId", "movieId", "label", "screen_year", "rating_counts", "rating_mean",
                                  "genres", "click_seq"],
                  "cat_columns": ["screen_year", "rating_counts", "rating_mean"],
                  "seq_columns": ["genres", "click_seq"],
                  "data_type": [np.str, np.str, np.float, np.str, np.str, np.str, np.str, np.str],
                  "default_val": [[""]] * 2 + [[0.0]] + [[""]] * 5,
                  "delimiter": ",",
                  "min_hit": 1,
                  "max_seq_num": 30,
                  "factor_dim": 50,
                  "batch_size": 128,
                  "learning_rate": 0.01,
                  "normal_mean": 0.0,
                  "normal_stddev": 1.0,
                  "epoch_num": 1}
    if os.path.exists(param_dict["model_dir"]):
        shutil.rmtree(param_dict["model_dir"])
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=run)
