# Author: Hayden Lao
# Script Name: get_train_data
# Created Date: Sep 29th 2020
# Description: Construct train data for FM model

import re
import numpy as np
import pandas as pd
from datetime import datetime


def get_movies_profile(data_file):
    df = pd.read_csv(data_file)
    screen_year_list = []
    for title in df["title"]:
        year_raw = re.findall(r'(\d{4})', title)
        screen_year = int(year_raw[0]) if len(year_raw) != 0 else 0
        current_year = datetime.now().year
        if screen_year == 0:
            screen_year_list.append("0")
        elif current_year - screen_year == 0:
            screen_year_list.append("1")
        elif current_year - screen_year <= 1:
            screen_year_list.append("2")
        elif current_year - screen_year <= 3:
            screen_year_list.append("3")
        elif current_year - screen_year <= 5:
            screen_year_list.append("4")
        elif current_year - screen_year <= 10:
            screen_year_list.append("5")
        elif current_year - screen_year <= 20:
            screen_year_list.append("6")
        elif current_year - screen_year <= 30:
            screen_year_list.append("7")
        else:
            screen_year_list.append("8")
    df["screen_year"] = screen_year_list
    return df


def get_rating_feats(data_file):
    g = pd.read_csv(data_file).groupby("movieId")
    rating_counts = g.agg({"userId": pd.Series.nunique}).rename(columns={"userId": "rating_counts"})
    rating_mean = g.agg({"rating": pd.Series.mean}).rename(columns={"rating": "rating_mean"})
    rating_feats = rating_counts.merge(rating_mean, how="inner", on="movieId")
    rating_feats["rating_counts"] = pd.qcut(rating_feats["rating_counts"].rank(method="first"),
                                            q=[0., 0.2, 0.4, 0.6, 0.8, 1],
                                            labels=list(map(lambda x: str(x), range(1, 6))))
    rating_mean_list = []
    for i in rating_feats["rating_mean"]:
        if i < 1:
            rating_mean_list.append("0")
            continue
        if i < 2:
            rating_mean_list.append("1")
            continue
        if i < 3:
            rating_mean_list.append("2")
            continue
        if i < 4:
            rating_mean_list.append("3")
            continue
        if i <= 5:
            rating_mean_list.append("4")
            continue
    rating_feats["rating_mean"] = rating_mean_list
    return rating_feats


def get_click_seq(click_df):
    g = click_df.groupby("userId")
    click_seq = g.apply(lambda x: process_click_seq_info(x)).reset_index()
    click_seq.columns = ["userId", "click_seq_info"]
    return click_seq


def process_click_seq_info(g):
    g = g.sort_values("timestamp", ascending=False)
    return "|".join(g["movieId"].astype(str) + "," + g["timestamp"].astype(str))


def process_click_seq(x, max_seq_num):
    click_seq_info, timestamp = x["click_seq_info"], x["timestamp"]
    if type(click_seq_info) != str and np.isnan(click_seq_info):
        return ""
    if type(click_seq_info)==float:
        print(click_seq_info)
    clicks = [i.split(",")[0] for i in click_seq_info.split("|") if int(i.split(",")[1]) < timestamp][:max_seq_num]
    return "|".join(clicks)


def get_clean_df(movie_file, rating_file, threshold=4, max_seq_num=30):
    base_df = pd.read_csv(rating_file)
    base_df["label"] = base_df["rating"].apply(lambda x: 1 if x >= threshold else 0)
    movie_df = get_movies_profile(movie_file)
    rating_feats = get_rating_feats(rating_file)
    click_seq = get_click_seq(base_df[base_df["label"] == 1])
    clean_df = base_df.merge(movie_df, how="left", on="movieId") \
        .merge(rating_feats, how="left", on="movieId") \
        .merge(click_seq, how="left", on="userId")
    clean_df["click_seq"] = clean_df.apply(lambda x: process_click_seq(x, max_seq_num), axis=1)
    return clean_df

if __name__ == '__main__':
    movie_file = "../data/movieLens25m_201912/movies.csv"
    rating_file = "../data/movieLens25m_201912/ratings.csv"
    target_file = "../data/movieLens25m_201912/data.csv"
    threshold = 4
    max_seq_num = 30
    columns = ["userId", "movieId", "label", "rating", "screen_year", "rating_counts", "rating_mean", "genres",
               "click_seq"]
    clean_df = get_clean_df(movie_file, rating_file, threshold, max_seq_num)[columns]
    # pd.set_option("display.max_columns",100)
    # print(clean_df.head())
    clean_df.to_csv(target_file, index=False)
