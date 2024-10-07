import re
import math
import pandas as pd
from collections import defaultdict


# numeric segment
def numeric_seg(x, bin_list):
    bin_list = sorted(bin_list)
    for i, bin_num in enumerate(bin_list):
        if x <= bin_num:
            return i


# publish year
def retrieve_publish_year(x):
    if re.match(r"\(\d{4}\)$", x.strip()[-6:]) is None:
        publish_year = int(x.strip()[-5:-1])
        return str(publish_year)
    else:
        return "unknown"


# title
def normalize_title(x):
    if re.match(r"\(\d{4}\)", x.strip()[-6:]):
        return x[:-6].strip().lower()
    else:
        return x.strip().lower()


# tag
def normalize_tag(x):
    if isinstance(x, str):
        return x.strip().lower()
    else:
        return "unknown"


# genres
def normalize_genres(x):
    if isinstance(x, str):
        genre_list = x.strip().lower().split("|")
        genre_list = sorted(genre_list)
        return "|".join(genre_list)
    else:
        return "unknown"


# click sequence
def get_click_seq_info(df, positive_rating):
    user_group = df[df["rating"] >= positive_rating].groupby("userId")
    user_list, click_seq_info_list = [], []
    for user_id, click_history in user_group:
        click_history = click_history.sort_values("timestamp", ascending=False)
        click_seq_info = "|".join(click_history["movieId"].astype(str) + "," + click_history["timestamp"].astype(str))
        click_seq_info_list.append(click_seq_info)
        user_list.append(user_id)
    click_seq_info = dict(zip(user_list, click_seq_info_list))
    return click_seq_info


def get_click_seq(x, click_seq_info, max_click_length):
    user_id, timestamp = x["userId"], x["timestamp"]
    click_seq = click_seq_info.get(user_id, ",1").split("|")
    clicks = [i.split(",")[0] for i in click_seq if int(i.split(",")[1]) < timestamp][:max_click_length]
    return "|".join(clicks)


# movie rating users
def get_movie_rating_users(ratings_df, bin_list=(
        100, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 15000, 50000, 100000)):
    movie_group = ratings_df.groupby("movieId")
    movie_list, rating_users = [], []
    for movie_id, ratings_info in movie_group:
        user_count = ratings_info.userId.unique().size
        user_bin = numeric_seg(user_count, bin_list)
        movie_list.append(movie_id)
        rating_users.append(user_bin)
    movie_rating_users = dict(zip(movie_list, rating_users))
    return movie_rating_users


# movie average rating
def get_movie_average_rating(ratings_df, bin_list=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)):
    movie_group = ratings_df.groupby("movieId")
    movie_list, average_rating_list = [], []
    for movie_id, ratings_info in movie_group:
        average_rating = ratings_info.rating.mean()
        average_rating_bin = numeric_seg(average_rating, bin_list)
        movie_list.append(movie_id)
        average_rating_list.append(average_rating_bin)
    movie_average_rating = dict(zip(movie_list, average_rating_list))
    return movie_average_rating


# user rating movies
def get_user_rating_movies(ratings_df, bin_list=(3, 5, 10, 20, 50, 70, 80, 100, 200, 500, 1000, 5000)):
    user_group = ratings_df.groupby("userId")
    user_list, rating_movies = [], []
    for user_id, ratings_info in user_group:
        movie_count = ratings_info.movieId.unique().size
        movie_bin = numeric_seg(movie_count, bin_list)
        user_list.append(user_id)
        rating_movies.append(movie_bin)
    user_rating_movies = dict(zip(user_list, rating_movies))
    return user_rating_movies


# user average rating
def get_user_average_rating(ratings_df, bin_list=(0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)):
    user_group = ratings_df.groupby("userId")
    user_list, average_rating_list = [], []
    for user_id, ratings_info in user_group:
        average_rating = ratings_info.rating.mean()
        average_rating_bin = numeric_seg(average_rating, bin_list)
        user_list.append(user_id)
        average_rating_list.append(average_rating_bin)
    user_average_rating = dict(zip(user_list, average_rating_list))
    return user_average_rating


# user most rated genre
def get_user_most_rated_genre(instance_df, min_rated_num):
    user_group = instance_df.groupby("userId")
    user_list, most_rated_genre_list = [], []
    for user_id, genres_info in user_group:
        genres_dict = defaultdict(int)
        genres_series = genres_info.genres.apply(lambda x: x.split("|"))
        for genres_list in genres_series:
            for genre in genres_list:
                genres_dict[genre] += 1
        genre, rated_num = sorted(list(genres_dict.items()), key=lambda x: x[1], reverse=True)[0]
        if rated_num < min_rated_num:
            genre = "not enough"
        user_list.append(user_id)
        most_rated_genre_list.append(genre)
    user_most_rated_genre = dict(zip(user_list, most_rated_genre_list))
    return user_most_rated_genre


# user favourite genre
def get_user_highest_rated_genre(instance_df, min_rated_num):
    user_group = instance_df.groupby("userId")
    user_list, favourite_genre_list = [], []
    for user_id, genres_info in user_group:
        genres_dict = defaultdict(list)
        genres_series = genres_info[["genres", "rating"]].apply(lambda x: (x[0].split("|"), x[1]), axis=1)
        for genres_list, rating in genres_series:
            rating = float(rating)
            for genre in genres_list:
                if genre in genres_dict:
                    genres_dict[genre][0] += 1
                    genres_dict[genre][1] += rating
                else:
                    genres_dict[genre] = [1, rating]
        genres_dict = {k: v[1] / v[0] for k, v in genres_dict.items() if v[0] >= min_rated_num}
        if len(genres_dict) == 0:
            genre = "not enough"
        else:
            genre, _ = sorted(list(genres_dict.items()), key=lambda x: x[1], reverse=True)[0]
        user_list.append(user_id)
        favourite_genre_list.append(genre)
    user_favourite_genre = dict(zip(user_list, favourite_genre_list))
    return user_favourite_genre


# rating counts
def rating_counts(df, group_feat, count_feat, bin_num=10):
    ratings_group = df.groupby(group_feat)[count_feat].nunique()
    bin_values = log_ef_cpt(ratings_group, bin_num=bin_num)
    group_list = ratings_group.keys()
    group_rating_counts = dict(zip(group_list, bin_values.codes.astype(str)))
    return group_rating_counts


# rating average
def rating_average(df, group_feat, count_feat, bin_num=10):
    ratings_group = df.groupby(group_feat)[count_feat].mean()
    bin_values = log_ef_cpt(ratings_group, bin_num=bin_num)
    group_list = ratings_group.keys()
    group_rating_counts = dict(zip(group_list, bin_values.codes.astype(str)))
    return group_rating_counts


def log_ef_cpt(feat, multiplier=1, offset=10e-7, log_base=math.e, bin_num=10, ceiling=None):
    if ceiling is not None:
        feat = feat.apply(lambda x: ceiling if x > ceiling else x)
    feat = feat.apply(lambda x: math.log(multiplier * x + offset, log_base))
    res = pd.qcut(feat, bin_num, duplicates="drop").values if isinstance(bin_num, int) \
        else pd.cut(feat, bin_num, duplicates="drop").values
    return res


def log_ew_cpt(feat, multiplier=1, offset=10e-7, log_base=math.e, bin_num=10, ceiling=None):
    if ceiling is not None:
        feat = feat.apply(lambda x: ceiling if x > ceiling else x)
    feat = feat.apply(lambda x: math.log(multiplier * float(x) + offset, log_base))
    res = pd.cut(feat, bin_num, duplicates="drop").values
    return res


def direct_cpt(feat, ceiling=None):
    if ceiling is not None:
        feat = feat.apply(lambda x: ceiling if x > ceiling else x)
    return feat


def combine_cpt(feats):
    return feats.apply(lambda row: "_".join(row.map(str)), axis=1)


def get_mean(feats, topk):
    input_list = [float(i) for i in feats.split(",") if float(i) >= 0][:topk]
    if sum(input_list) == 0:
        return 0
    return sum(input_list) / len(input_list)


def get_length(feats, topk):
    input_list = [float(i) for i in feats.split(",") if float(i) >= 0][:topk]
    return len(input_list)
