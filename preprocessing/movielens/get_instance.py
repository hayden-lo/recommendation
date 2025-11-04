import time
import pickle
from datetime import datetime
import env.configuration as conf
from preprocessing.movielens.computer import *
from utils.log_utils import logger
from utils.time_utils import seconds_elapse


def main():
    logger("Loading data")
    ratings_df = pd.read_csv(conf.MOVIELENS_25M_RATINGS)
    movies_df = pd.read_csv(conf.MOVIELENS_25M_MOVIES)

    logger("Merging data")
    merge_start = time.time()
    data_df = pd.merge(ratings_df, movies_df, on=["movieId"], how="left")
    logger(f"Merge data elapsed {seconds_elapse(merge_start)} seconds, data count: {data_df.shape[0]}")

    logger("Extracting features")
    extract_start = time.time()
    # label
    data_df["label"] = (data_df["rating"] >= conf.MOVIELENS_POSITIVE_THRESHOLD).astype(float)
    # weight
    data_df["weight"] = 1.0
    # publish year
    data_df["publish_year"] = data_df["title"].apply(retrieve_publish_year)
    # genres
    data_df["genres"] = data_df["genres"].apply(normalize_genres)
    # click sequence
    data_df["click_seq"] = get_click_seq(data_df, conf.MOVIELENS_MAX_CLICK_LENGTH)
    # movie rating users
    data_df["movie_rating_users"] = group_cumcount(data_df, "movieId", "userId", "timestamp", is_unique=True)
    # user rating movies
    data_df["user_rating_movies"] = group_cumcount(data_df, "userId", "movieId", "timestamp", is_unique=True)
    # movie sum rating
    data_df["movie_sum_rating"] = group_cumsum(data_df, "movieId", "rating", "timestamp")
    # user sum rating
    data_df["user_sum_rating"] = group_cumsum(data_df, "userId", "rating", "timestamp")
    # movie average rating
    data_df["movie_average_rating"] = data_df["movie_sum_rating"] / data_df["movie_rating_users"]
    # user average rating
    data_df["user_average_rating"] = data_df["user_sum_rating"] / data_df["user_rating_movies"]
    # user most rated genre
    data_df["user_most_rated_genre"] = get_group_most_rated_genre(data_df, "userId", "genres", "timestamp")
    # user favourite genre
    data_df["user_favourite_genre"] = get_group_highest_rated_genre(data_df, "userId", "genres", "rating", "timestamp")
    logger(f"Extract features elapsed {seconds_elapse(extract_start)} seconds")

    logger("Splitting train set and test set")
    train_instance = data_df[data_df["timestamp"] <= datetime.strptime(conf.MOVIELENS_TRAIN_TEST_SPLIT_DATE, "%Y-%m-%d").timestamp()]
    test_instance = data_df[data_df["timestamp"] > datetime.strptime(conf.MOVIELENS_TRAIN_TEST_SPLIT_DATE, "%Y-%m-%d").timestamp()]

    logger("Writing instance")
    instance_start = time.time()
    train_instance.to_csv(conf.MOVIELENS_TRAIN_DATA_PATH, index=False, mode="w")
    test_instance.to_csv(conf.MOVIELENS_TEST_DATA_PATH, index=False, mode="w")
    logger(f"Write instance elapsed {seconds_elapse(instance_start)} seconds")

    # logger("Writing vocab")
    # vocab_start = time.time()
    # vocab_dict = get_vocab_dict(train_instance, conf.SPARSE_FEATURES, conf.SEQUENCE_FEATURES, conf.SEQUENCE_DELIMITER)
    # f = open(conf.VOCAB_FILE, "wb")
    # pickle.dump(vocab_dict, f)
    # f.close()
    # logger(f"Write vocab elapsed {round((time.time() - vocab_start) / 60, 2)} mins")


if __name__ == '__main__':
    main()
