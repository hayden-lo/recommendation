# Author: Hayden Lao
# Script Name: main_item2vec
# Created Date: Mar 8th 2020
# Description: Main function for item2vec recommendation

from models.item2vec.model_item2vec_gensim import Item2VecGensim
from models.item2vec.iter_constructor import IterConstructor
from models.item2vec.get_train_data import produce_train_data


def train_model(**params):
    """
    LFM model train method
    :param params: Run parameters[dict]
    :return: Train LFM model, Nothing return
    """
    # Materialize lfm model
    lfm_model = LFM(**params)
    # Train and output both user and item vector files
    lfm_model.model_train_process(params["input_file"], params["uv_file"], params["iv_file"])


if __name__ == '__main__':
    # Run parameters
    params_dict = {"act_file": "data/movieLens1m_201809/ratings.csv",  # action logs file
                   "act_seq_file": "item2vec/vecData/act_seq.txt",  # action sequence file
                   "log_separator": ",",  # log field separator
                   "act_seq_separator": " ",  # action sequence separator
                   "pos_thr": 4,  # threshold score to be marked as positive
                   "max_seq": 30,  # maximum action sequence length
                   "gensim_model_file": "item2vec/gensim_w2v_model",  # gensim output word2vec model file
                   "gensim_vectors_file": "item2vec/vecData/gensim_vectors.txt",  # gensim output vectors file
                   "learning_rate": 0.01,  # learning rate
                   "batch_size": 128,  # batch size
                   "emb_size": 128,  # vector dimension
                   "window": 3,  # slide window
                   "min_count": 3,  # minimum action number
                   "network_algo": 1,  # network algorithm: 0 for CBOW, 1 for skip-gram
                   "speed_algo": 0,  # speed algorithm: 0 for negative sampling, 1 for hierarchical softmax
                   "negative_number": 10  # negative sampling number
                   }
    # Generate action sequence file
    produce_train_data(params_dict["act_file"], params_dict["act_seq_file"], params_dict["log_separator"],
                       params_dict["pos_thr"], params_dict["max_seq"])
    # Gensim model
    # Get action sequence iterator
    bow = IterConstructor(params_dict["act_seq_file"])
    # Train  model
    item2vecGensim = Item2VecGensim(**params_dict)
    gensim_word2vec_model = item2vecGensim.init_model(bow)
    # Get top n most similar items
    item2vecGensim.topn_similar()
