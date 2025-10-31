import pickle
from datetime import datetime
import preprocessing.movielens.preprocess_config as conf
from preprocessing.movielens.bak.computer import *
from utils.tf_utils import get_vocab_dict


def main():
    # Load data
    ratings_df = pd.read_csv(conf.RATINGS_FILE)
    movies_df = pd.read_csv(conf.MOVIES_FILE)
    tags_df = pd.read_csv(conf.TAGS_FILE)

    # Merge data2
    tags_df = tags_df.groupby(["userId", "movieId"])["tag"].unique().apply(
        lambda x: "|".join([str(i) for i in x])).reset_index()
    instance_df = pd.merge(ratings_df, movies_df, on=["movieId"], how="left")
    instance_df = pd.merge(instance_df, tags_df, on=["userId", "movieId"], how="left")
    # instance_df = instance_df.head(200)

    # Generate new fields
    # label
    instance_df["label"] = instance_df["rating"].apply(lambda x: 1.0 if x >= conf.POSITIVE_RATING else 0.0)
    # publish year
    instance_df["publish_year"] = instance_df["title"].apply(retrieve_publish_year)
    # title
    instance_df["title"] = instance_df["title"].apply(normalize_title)
    # tag
    instance_df["tag"] = instance_df["tag"].apply(normalize_tag)
    # genres
    instance_df["genres"] = instance_df["genres"].apply(normalize_genres)
    # click_seq
    click_seq_info = get_click_seq_info(ratings_df)
    instance_df["click_seq"] = instance_df.apply(lambda x: get_click_seq(x, click_seq_info, conf.MAX_CLICK_LENGTH),
                                                 axis=1)
    # movie rating users
    movie_rating_users = get_movie_rating_users(ratings_df)
    instance_df["movie_rating_users"] = instance_df.apply(lambda x: movie_rating_users[x.movieId], axis=1)
    # movie average rating
    movie_average_rating = get_movie_average_rating(ratings_df)
    instance_df["movie_average_rating"] = instance_df.apply(lambda x: movie_average_rating[x.movieId], axis=1)
    # user rating movies
    user_rating_movies = get_user_rating_movies(ratings_df)
    instance_df["user_rating_movies"] = instance_df.apply(lambda x: user_rating_movies[x.userId], axis=1)
    # user rating movies
    user_average_rating = get_user_average_rating(ratings_df)
    instance_df["user_average_rating"] = instance_df.apply(lambda x: user_average_rating[x.userId], axis=1)
    # user rating movies
    user_most_rated_genre = get_user_most_rated_genre(instance_df, conf.MIN_RATED_NUM)
    instance_df["user_most_rated_genre"] = instance_df.apply(lambda x: user_most_rated_genre[x.userId], axis=1)
    # user favourite genre
    user_favourite_genre = get_user_highest_rated_genre(instance_df, conf.MIN_RATED_NUM)
    instance_df["user_favourite_genre"] = instance_df.apply(lambda x: user_favourite_genre[x.userId], axis=1)

    # Split train and test data
    train_instance = instance_df[instance_df["timestamp"] <= datetime(2019, 10, 1).timestamp()]
    test_instance = instance_df[instance_df["timestamp"] > datetime(2019, 10, 1).timestamp()]

    # Write instance data
    train_instance.to_csv(conf.TRAIN_FILE, index=False, mode="w")
    test_instance.to_csv(conf.TEST_FILE, index=False, mode="w")

    # Write vocab dict
    vocab_dict = get_vocab_dict(train_instance, conf.SPARSE_FEATURES, conf.SEQUENCE_FEATURES, conf.SEQUENCE_DELIMITER)
    f = open(conf.VOCAB_FILE, "wb")
    pickle.dump(vocab_dict, f)
    f.close()


if __name__ == '__main__':
    main()
