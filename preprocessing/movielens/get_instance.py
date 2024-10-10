import time
import pickle
from datetime import datetime
import preprocessing.movielens.preprocess_config as conf
from preprocessing.movielens.computer import *
from utils.tf_utils import get_vocab_dict
from utils.log_utils import logger
from utils.toolkit import low_memory_df
from utils.time_utils import seconds_elapse


def main():
    logger("Loading data")
    ratings_df = pd.read_csv(conf.RATINGS_FILE)
    movies_df = pd.read_csv(conf.MOVIES_FILE)
    tags_df = pd.read_csv(conf.TAGS_FILE)

    logger("Merging data")
    merge_start = time.time()
    tags_df = tags_df.groupby(["userId", "movieId"])["tag"].unique().apply(
        lambda x: "|".join([str(i) for i in x])).reset_index()
    data_df = pd.merge(ratings_df, movies_df, on=["movieId"], how="left")
    data_df = pd.merge(data_df, tags_df, on=["userId", "movieId"], how="left")
    data_df = low_memory_df(data_df)
    logger(f"Merge data elapsed {round((time.time() - merge_start) / 60, 2)} mins, data count: {data_df.shape[0]}")

    logger("Extracting features")
    extract_start = time.time()
    # label
    data_df["label"] = data_df["rating"].apply(lambda x: 1.0 if x >= conf.POSITIVE_RATING else 0.0)
    # weight
    data_df["weight"] = 1.0
    # publish year
    data_df["publish_year"] = data_df["title"].apply(retrieve_publish_year)
    # title
    data_df["title"] = data_df["title"].apply(normalize_title)
    # tag
    data_df["tag"] = data_df["tag"].apply(normalize_tag)
    # genres
    data_df["genres"] = data_df["genres"].apply(normalize_genres)
    logger(f"genres, {seconds_elapse(extract_start)}")
    # click sequence
    click_seq_info = get_click_seq_info(data_df, conf.POSITIVE_RATING)
    data_df["click_seq"] = data_df.apply(lambda x: get_click_seq(x, click_seq_info, conf.MAX_CLICK_LENGTH), axis=1)
    logger(f"click sequence, {seconds_elapse(extract_start)}")
    # movie rating users
    data_df["movie_rating_users"] = group_cumcount(data_df, "movieId", "userId", "timestamp", is_unique=True)
    logger(f"movie_rating_users, {seconds_elapse(extract_start)}")
    # user rating movies
    data_df["user_rating_movies"] = group_cumcount(data_df, "userId", "movieId", "timestamp", is_unique=True)
    logger(f"user_rating_movies, {seconds_elapse(extract_start)}")
    # movie sum rating
    data_df["movie_sum_rating"] = group_cumsum(data_df, "movieId", "rating", "timestamp")
    logger(f"movie_sum_rating, {seconds_elapse(extract_start)}")
    # user sum rating
    data_df["user_sum_rating"] = group_cumsum(data_df, "userId", "rating", "timestamp")
    logger(f"user_sum_rating, {seconds_elapse(extract_start)}")
    # movie average rating
    data_df["movie_average_rating"] = data_df["movie_sum_rating"] / data_df["movie_rating_users"]
    logger(f"movie_average_rating, {seconds_elapse(extract_start)}")
    # user average rating
    data_df["user_average_rating"] = data_df["user_sum_rating"] / data_df["user_rating_movies"]
    logger(f"user_average_rating, {seconds_elapse(extract_start)}")
    # user most rated genre
    data_df["user_most_rated_genre"] = get_group_most_rated_genre(data_df, "userId", "genres", "timestamp")
    logger(f"user_most_rated_genre, {seconds_elapse(extract_start)}")
    # user favourite genre
    data_df["user_favourite_genre"] = get_group_highest_rated_genre(data_df, "userId", "genres", "rating", "timestamp")
    logger(f"user_favourite_genre, {seconds_elapse(extract_start)}")
    logger(f"Extract features elapsed {round((time.time() - extract_start) / 60, 2)} mins")

    # import sys
    # sys.exit(-1)

    logger("Splitting train set and test set")
    train_instance = data_df[data_df["timestamp"] <= datetime.strptime(conf.SPLIT_DATE, "%Y-%m-%d").timestamp()]
    test_instance = data_df[data_df["timestamp"] > datetime.strptime(conf.SPLIT_DATE, "%Y-%m-%d").timestamp()]

    logger("Writing instance")
    instance_start = time.time()
    train_instance.to_csv(conf.TRAIN_FILE, index=False, mode="w")
    test_instance.to_csv(conf.TEST_FILE, index=False, mode="w")
    logger(f"Write instance elapsed {round((time.time() - instance_start) / 60, 2)} mins")

    logger("Writing vocab")
    vocab_start = time.time()
    vocab_dict = get_vocab_dict(train_instance, conf.SPARSE_FEATURES, conf.SEQUENCE_FEATURES, conf.SEQUENCE_DELIMITER)
    f = open(conf.VOCAB_FILE, "wb")
    pickle.dump(vocab_dict, f)
    f.close()
    logger(f"Write vocab elapsed {round((time.time() - vocab_start) / 60, 2)} mins")


if __name__ == '__main__':
    main()
