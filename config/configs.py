import numpy as np

universal_params = {"mode": "dev",
                    "model_dir": "./model_dir",
                    "data_dir": "../../data/movieLens1m_201809",
                    "data_file": "data.csv",
                    "train_file": "train.csv",
                    "test_file": "test.csv",
                    "all_columns": ["userId", "movieId", "label", "rating", "screen_year", "rating_counts",
                                    "rating_mean", "genres", "click_seq"],
                    "cat_columns": ["screen_year", "rating_counts", "rating_mean"],
                    "seq_columns": {"genres": 5, "click_seq": 30},
                    "data_type": [np.str, np.str, np.float, np.float, np.str, np.str, np.str, np.str, np.str],
                    "default_val": [[""]] * 2 + [[0.0]] * 2 + [[""]] * 5,
                    "delimiter": ",",
                    "padding_value": "padding_value",
                    "min_hit": 3,
                    "reg_fun": "l2",
                    "batch_size": 128,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "loss_fun": "binary_cross_entropy",
                    "from_logits": False,
                    "metrics": ["auc"],
                    "epoch_num": 1}

train_params = {"mode": "train",
                "model_dir": "./model_dir_big",
                "data_dir": "../../data/movieLens25m_201912"}

predict_params = {"mode": "predict",
                  "model_dir": "./model_dir_big",
                  "data_dir": "../../data/movieLens25m_201912",
                  "predict_file": "predict_info.csv",
                  "user_features": ["click_seq", "movie_id"],
                  "item_features": ["movieId", "screen_year", "rating_counts", "rating_mean", "genres"]}