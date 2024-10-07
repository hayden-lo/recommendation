import re
from utils.bak.get_inputs import *


def get_recommendations(inputs, outputs, param_dict):
    candidates = [i.replace("click_seq_", "") for i in inputs["movieId"].squeeze().tolist()]
    results = list(zip(candidates, outputs.tolist()))
    results_df = pd.DataFrame(sorted(results, key=lambda x: x[1], reverse=True), columns=["movieId", "score"])
    results_df["movieId"] = results_df["movieId"].astype(np.int64)
    movies_df = pd.read_csv(os.path.join(param_dict["data_dir"], "movies.csv"))
    rating_df = pd.read_csv(os.path.join(param_dict["data_dir"], "ratings.csv"))
    rating_counts = rating_df.groupby("movieId").agg({"userId": pd.Series.nunique}).rename(
        columns={"userId": "rating_counts"})
    movies_df["screen_year"] = movies_df["title"].apply(
        lambda x: int(re.findall(r'(\d{4})', x)[0]) if len(re.findall(r'(\d{4})', x)) != 0 else 9999)
    movieId2imdb = pd.DataFrame([[v[0], k] for k, v in imdb2id(param_dict).items()], columns=["movieId", "imdb_id"])
    recom_df = results_df.merge(movies_df, how="inner", on="movieId") \
        .merge(movieId2imdb, how="inner", on="movieId") \
        .merge(rating_counts, how="inner", on="movieId")
    return recom_df[["movieId", "imdb_id", "score", "title", "screen_year", "genres", "rating_counts"]]
