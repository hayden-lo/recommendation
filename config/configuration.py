import numpy as np

# Instance parameters
RATINGS_FILE = "D:\\git\\recommendation\\data\\ml-25m\\ratings.csv"
MOVIES_FILE = "D:\\git\\recommendation\\data\\ml-25m\\movies.csv"
TAGS_FILE = "D:\\git\\recommendation\\data\\ml-25m\\tags.csv"
INSTANCE_FILE = "D:\\git\\recommendation\\data\\ml-25m\\instance.csv"
TRAIN_FILE = "D:\\git\\recommendation\\data\\ml-25m\\train.csv"
TEST_FILE = "D:\\git\\recommendation\\data\\ml-25m\\test.csv"
MAX_CLICK_LENGTH = 10
MIN_RATED_NUM = 10
POSITIVE_RATING = 4.0

# Default parameters
FIELD_DELIMITER = ","
PADDING_VALUE = "padding_value"
MIN_HIT = 3
EMBEDDING_SIZE = 128
ACTIVATION_FUNCTION = "relu"
REGULARIZATION_FUNCTION = "l2"
BATCH_SIZE = 128

# Model debug inputs
debug_inputs = {"movieId": np.array([["click_seq_157"]]),
                "screen_year": np.array([["screen_year_7"]]),
                "rating_counts": np.array([["rating_counts_4"]]),
                "rating_mean": np.array([["rating_mean_2"]]),
                "click_seq": np.array([["click_seq_2492", "click_seq_2012", "click_seq_2478", "click_seq_553",
                                        "click_seq_157", "click_seq_3053", "click_seq_1298", "click_seq_3448",
                                        "click_seq_151", "click_seq_1090", "click_seq_1224", "click_seq_5060",
                                        "click_seq_527", "click_seq_3147", "click_seq_2353", "click_seq_47",
                                        "click_seq_593", "click_seq_3033", "click_seq_1206", "click_seq_3702",
                                        "click_seq_1240", "click_seq_1270", "click_seq_2291", "click_seq_163",
                                        "click_seq_1226", "click_seq_943", "click_seq_1265", "click_seq_3273",
                                        "click_seq_1625", "click_seq_1092"]]),
                "genres": np.array([["genres_Comedy", "genres_War"] + ["padding_value"] * 8])}
