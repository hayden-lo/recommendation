# Instance parameters
RATINGS_FILE = "C:\\git\\recommendation\\data\\ml-32m\\ml-32m\\ratings.csv"
MOVIES_FILE = "C:\\git\\recommendation\\data\\ml-32m\\ml-32m\\movies.csv"
TAGS_FILE = "C:\\git\\recommendation\\data\\ml-32m\\ml-32m\\tags.csv"
TRAIN_FILE = "C:\\git\\recommendation\\data\\ml-32m\\ml-32m\\train.csv"
TEST_FILE = "C:\\git\\recommendation\\data\\ml-32m\\ml-32m\\test.csv"
VOCAB_FILE = "C:\\git\\recommendation\\data\\ml-32m\\ml-32m\\vocab_dict.pkl"
SPLIT_DATE = "2019-10-01"
MAX_CLICK_LENGTH = 10
MIN_RATED_NUM = 10
POSITIVE_RATING = 4.0
SEQUENCE_DELIMITER = "|"
OUT_SCHEMA = ["userId", "movieId", "timestamp", "rating", "weight", "label"]
SPARSE_FEATURES = ["publish_year", "user_rating_movies", "movie_rating_users", "user_average_rating",
                   "movie_average_rating", "user_most_rated_genre", "user_favourite_genre"]
SEQUENCE_FEATURES = {"click_seq": 10, "genres": 3}
