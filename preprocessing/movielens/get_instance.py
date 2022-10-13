import pandas as pd

RATINGS_FILE = "D:\\git\\recommendation\\data\\ml-25m\\ml-25m\\ratings.csv"
MOVIES_FILE = "D:\\git\\recommendation\\data\\ml-25m\\ml-25m\\movies.csv"
TAGS_FILE = "D:\\git\\recommendation\\data\\ml-25m\\ml-25m\\tags.csv"
DATA_FLOW_FILE = "D:\\git\\recommendation\\data\\ml-25m\\ml-25m\\data_flow.csv"

# Load data
ratings_df = pd.read_csv(RATINGS_FILE)
movies_df = pd.read_csv(MOVIES_FILE)
tags_df = pd.read_csv(TAGS_FILE)

# Merge data
data_flow_df = pd.merge(ratings_df, movies_df, on=["movieId"], how="left")
data_flow_df = pd.merge(data_flow_df, tags_df[["userId", "movieId", "tag"]], on=["userId", "movieId"], how="left")

# Generate new fields
data_flow_df[""]

# Write data flow
data_flow_df.head(100).to_csv(DATA_FLOW_FILE, index=False, mode="")
