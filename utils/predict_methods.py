import os
import pandas as pd
from utils.get_inputs import *


def get_recommendations(inputs, outputs, param_dict):
    candidates = [i.replace("click_seq_", "") for i in inputs["movieId"].squeeze().tolist()]
    results = list(zip(candidates, outputs.tolist()))
    results_df = pd.DataFrame(sorted(results, key=lambda x: x[1], reverse=True), columns=["movieId", "score"])
    results_df["movieId"] = results_df["movieId"].astype(np.int64)
    movies_df = pd.read_csv(os.path.join(param_dict["data_dir"], "movies.csv"))
    movieId2imdb = pd.DataFrame([[v[0], k] for k, v in imdb2id(param_dict).items()], columns=["movieId", "imdb_id"])
    recom_df = results_df.merge(movies_df, how="inner", on="movieId").merge(movieId2imdb, how="inner", on="movieId")
    return recom_df[["movieId", "imdb_id", "score", "title","genres"]]
