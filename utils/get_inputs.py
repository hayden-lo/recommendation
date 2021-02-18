import os
import pandas as pd
import numpy as np
from collections import defaultdict


def get_inputs(user_id, param_dict):
    inputs_file = os.path.join(param_dict["data_dir"], param_dict["inputs_file"])
    if os.path.exists(inputs_file):
        f = open(inputs_file, "r")
        inputs_dict = eval(f.read().strip())
        if user_id in inputs_dict:
            return {k: np.array(v) for k, v in inputs_dict[user_id].items()}
        f.close()
    inputs = defaultdict(np.ndarray)
    all_data = pd.read_csv(os.path.join(param_dict["data_dir"], param_dict["data_file"]))
    user_data = pd.read_csv(os.path.join(param_dict["data_dir"], param_dict["predict_file"]))
    user_data = user_data[user_data["user_id"] == user_id]
    watched = list(user_data["movie_id"].unique())
    # item features
    item_features_df = all_data[param_dict["item_features"]].drop_duplicates(keep='first')
    item_features_df = item_features_df[~item_features_df["movieId"].isin(watched)]
    for k, v in item_features_df.to_dict().items():
        if k == "movieId":
            inputs[k] = np.array([["click_seq_" + str(i)] for i in v.values()])
        if k in param_dict["cat_columns"]:
            inputs[k] = np.array([[k + "_" + str(i)] for i in v.values()])
        if k in param_dict["seq_columns"]:
            array_list = []
            max_seq_num = param_dict["seq_columns"][k]
            for row in v.values():
                feat_list = [k + "_" + genre for genre in row.split("|")][:max_seq_num]
                if len(feat_list) < max_seq_num:
                    feat_list += [param_dict["padding_value"]] * (max_seq_num - len(feat_list))
                array_list.append(feat_list)
            inputs[k] = np.array(array_list)
    candidates = item_features_df.shape[0]
    # user features
    cols = list(set(user_data.columns).intersection(set(param_dict["user_features"])))
    user_features_df = user_data[cols].drop_duplicates(keep='first')
    for col in cols:
        if col == "movie_id":
            click_seq = user_data[user_data["ratings"] >= 4].sort_values("ratings", ascending=False)[
                "movie_id"].to_list()
            click_seq = ["click_seq_" + str(i) for i in click_seq]
            if len(click_seq) > 30:
                click_seq = np.random.choice(click_seq, size=30, replace=False)
            if len(click_seq) < 30:
                click_seq += [param_dict["padding_value"]] * (30 - len(click_seq))
            inputs["click_seq"] = np.array([click_seq] * candidates)
        if col in param_dict["cat_columns"]:
            inputs[col] = np.array([user_features_df[col]] * candidates)
        if col in param_dict["seq_columns"]:
            max_seq_num = param_dict["seq_columns"][col]
            feat_list = [col + "_" + i for i in user_features_df[col].item().split("|")][:max_seq_num]
            if len(feat_list) < max_seq_num:
                feat_list += [param_dict["padding_value"]] * (max_seq_num - len(feat_list))
            inputs[col] = np.array([feat_list] * candidates)
    if os.path.exists(inputs_file):
        f = open(inputs_file, "r")
        inputs_dict = eval(f.read().strip())
        inputs_dict = {**inputs_dict, **{user_id: {k: v.tolist() for k, v in dict(inputs).items()}}}
        f = open(inputs_file, "w")
        f.write(str(inputs_dict))
    else:
        f = open(inputs_file, "w")
        f.write(str({user_id: {k: v.tolist() for k, v in dict(inputs).items()}}))
    return dict(inputs)


def imdb2id(param_dict):
    movies_file = os.path.join(param_dict["data_dir"], "movies.csv")
    links_file = os.path.join(param_dict["data_dir"], "links.csv")
    movies_df = pd.read_csv(movies_file)
    links_df = pd.read_csv(links_file)
    merge_df = links_df.merge(movies_df, how="left", on="movieId")[["imdbId", "movieId", "title"]]
    imdb2id_dict = {v["imdbId"]: [v["movieId"], v["title"]] for v in merge_df.T.to_dict().values()}
    return imdb2id_dict


def user_inputs_glance(param_dict, user_id):
    user_data = pd.read_csv(os.path.join(param_dict["data_dir"], param_dict["predict_file"]))
    user_data = user_data[user_data["user_id"] == user_id]
    movies_df = pd.read_csv(os.path.join(param_dict["data_dir"], "movies.csv"))
    user_rating_df = user_data.merge(movies_df, how="inner", left_on="movie_id", right_on="movieId")
    user_rating_df = user_rating_df.sort_values("ratings", ascending=False)
    return user_rating_df[["movie_id", "ratings", "title", "genres"]]
