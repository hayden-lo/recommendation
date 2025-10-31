import re
import math
import pandas as pd
from itertools import chain
from collections import defaultdict, Counter


# publish year
def retrieve_publish_year(x, default_value="unknown"):
    if re.match(r"\((\d{4})\)$", x.strip()[-6:]) is not None:
        return x.strip()[-5:-1]
    else:
        return default_value


# title
def normalize_title(x):
    if re.match(r"\((\d{4})\)$", x.strip()[-6:]):
        return x[:-6].strip().lower()
    else:
        return x.strip().lower()


# genres
def normalize_genres(x, default_value="unknown"):
    if isinstance(x, str):
        genre_list = x.strip().lower().split("|")
        genre_list = sorted(genre_list)
        return "|".join(genre_list)
    else:
        return default_value


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


# def get_click_seq(x, click_seq_info, max_click_length):
#     user_id, timestamp = x["userId"], x["timestamp"]
#     click_seq = click_seq_info.get(user_id, ",1").split("|")
#     clicks = [i.split(",")[0] for i in click_seq if int(i.split(",")[1]) < timestamp][:max_click_length]
#     return "|".join(clicks)

def get_click_seq(df, max_click_length):
    df = df[df["label"] == 1].sort_values(by=["userId", "timestamp"])
    df["movie_ts"] = df["movieId"].astype(str) + "::" + df["timestamp"].astype(str)
    df["click_seq"] = df.groupby("userId")["movie_ts"].transform(
        lambda x: (x + ",").cumsum().shift(1).str.rstrip(","))
    df["click_seq"] = df["click_seq"].str.split(",").apply(lambda x: ",".join(x[-max_click_length:]))
    return df


# group counts
def group_count(df, group_feat, count_feat, is_unique=True):
    groups = df.groupby(group_feat)[count_feat]
    if is_unique:
        groups = groups.nunique()
    group_counts_dict = dict(groups)
    return df[group_feat].map(group_counts_dict)


# group cumulative count
def group_cumcount(df, group_feat, count_feat, time_feat, is_unique=True):
    need_df = df[[group_feat, count_feat, time_feat]]
    need_df = need_df.sort_values(by=[group_feat, time_feat])
    if is_unique:
        need_df = need_df.drop_duplicates(subset=[group_feat, count_feat]).sort_values(by=[group_feat, count_feat])
    return need_df.groupby(group_feat).cumcount()


# group sum
def group_sum(df, group_feat, avg_feat):
    groups = df.groupby(group_feat)[avg_feat].mean()
    group_counts_dict = dict(groups)
    return df[group_feat].map(group_counts_dict)


# group cumulative sum
def group_cumsum(df, group_feat, sum_feat, time_feat):
    need_df = df[[group_feat, sum_feat, time_feat]]
    need_df = need_df.sort_values(by=[group_feat, time_feat])
    return need_df.groupby(group_feat).apply(lambda x: x[sum_feat].shift(1).cumsum()).reset_index(level=0, drop=True)


def get_group_most_rated_genre(df, group_feat, tag_feat, time_feat):
    data_list = []
    df = df[[group_feat, tag_feat, time_feat]]
    group_expanding = df.sort_values(by=time_feat).groupby(group_feat).expanding()
    for expanding in group_expanding:
        genre_list = expanding.shift(1)[tag_feat].str.split("|").dropna().to_list()
        counter = Counter(chain.from_iterable(genre_list)).most_common(1) if len(genre_list) > 0 else [[""]]
        data_list.append(counter[0][0])
    return data_list


def get_group_highest_rated_genre(df, group_feat, tag_feat, rate_feat, time_feat):
    data_list = []
    df = df[[group_feat, tag_feat, rate_feat, time_feat]]
    group_expanding = df.sort_values(by=time_feat).groupby(group_feat).expanding()
    for expanding in group_expanding:
        expanding.loc[:, tag_feat] = expanding.shift(1)[tag_feat].str.split("|")
        expanding = expanding.dropna()
        genre_datas = zip(expanding[tag_feat], expanding[rate_feat].dropna())
        genre_count = defaultdict(int)
        genre_rate = defaultdict(float)
        for genre_data in genre_datas:
            if len(genre_data) == 0:
                continue
            for genre in genre_data[0]:
                genre_count[genre] += 1
                genre_rate[genre] += genre_data[1]
        genre_avg_rate = {k: genre_rate[k] / v for k, v in genre_count.items()}
        data_list.append(max(genre_avg_rate, key=genre_avg_rate.get) if len(genre_avg_rate) > 0 else "")
    return data_list


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
