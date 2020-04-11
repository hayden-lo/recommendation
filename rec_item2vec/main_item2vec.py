# Author: Hayden Lao
# Script Name: main_item2vec
# Created Date: Mar 8th 2020
# Description: Main function for item2vec recommendation

from rec_item2vec.model_item2vec_gensim import Item2VecGensim
from rec_item2vec.get_train_data import produce_train_data

#
produce_train_data("recData/movieLens1m_201809/ratings.csv","rec_item2vec/vecData/act_seq.txt")

if __name__ == '__main__':
    # Run parameters
    params_dict = {"latent_factor": 50, "alpha": 0.01, "learning_rate": 0.01, "step": 10, "threshold": 4,
                   "input_file": "../recData/movieLens25m_201912/ratings.csv", "uv_file": "vecData/user_vec.txt",
                   "iv_file": "vecData/item_vec.txt", "movie_info": "../recData/movieLens25m_201912/movies.csv", "top_n": 10}
    # Train lfm model
    train_model(**params_dict)
    rec_analysis("999999", **params_dict)
