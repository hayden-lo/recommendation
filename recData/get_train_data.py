# Author: Hayden Lao
# Script Name: get_train_data
# Created Date: Sep 29th 2020
# Description: Construct train data for FM model

import re
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


def get_click_seq(click_df, max_seq_num):
    g = click_df.groupby("userId")
    click_seq = g.apply(lambda x: "|".join(
        x.sort_values("timestamp", ascending=False).head(max_seq_num)["movieId"].astype(str))).reset_index()
    click_seq.columns = ["userId", "click_seq"]
    return click_seq


def get_clean_df(movie_file, rating_file, threshold=4, max_seq_num=30):
    base_df = pd.read_csv(rating_file)
    base_df["label"] = base_df["rating"].apply(lambda x: 1 if x >= threshold else 0)
    movie_df = get_movies_profile(movie_file)
    rating_feats = get_rating_feats(rating_file)
    click_seq = get_click_seq(base_df[base_df["label"] == 1], max_seq_num)
    clean_df = base_df.merge(movie_df, how="left", on="movieId") \
        .merge(rating_feats, how="left", on="movieId") \
        .merge(click_seq, how="left", on="userId")
    return clean_df


if __name__ == '__main__':
    movie_file = "movieLens1m_201809/movies.csv"
    rating_file = "movieLens1m_201809/ratings.csv"
    target_file = "movieLens1m_201809/data.csv"
    threshold = 4
    max_seq_num = 30
    columns = ["userId", "movieId", "label", "rating", "screen_year", "rating_counts", "rating_mean", "genres",
               "click_seq"]
    clean_df = get_clean_df(movie_file, rating_file, threshold, max_seq_num)[columns]
    # pd.set_option("display.max_columns",100)
    # print(clean_df.head())
    clean_df.to_csv(target_file, index=False)
