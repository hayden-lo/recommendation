# Author: Hayden Lao
# Script Name: main_lfm
# Created Date: Sep 5th 2019
# Description: Main function for latent factor model recommendation

from models.lfm.model_lfm import LFM
from models.lfm.get_train_data import get_train_data
from utils.bak.readDataUtils import get_movie_info


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


def get_vectors(uv_file, iv_file):
    """
    Read vector into dictionary from written files
    :param uv_file: User vector file[str]
    :param iv_file: Item vector file[str]
    :return: A tuple contains user vector dictionary and item vector dictionary[tuple]
    """
    user_vector = {}
    with open(uv_file) as user_file:
        for user in user_file:
            user_id, vec_list = user.split()[0], [float(v) for v in user.split()[1].split(",")]
            user_vector[user_id] = vec_list
    item_vector = {}
    with open(iv_file) as item_file:
        for item in item_file:
            item_id, vec_list = item.split()[0], [float(v) for v in item.split()[1].split(",")]
            item_vector[item_id] = vec_list
    return user_vector, item_vector


def rec_top_n(user_vec, item_vec, user_id, top_n=10):
    """
    Return top n recommendation list
    :param user_vec: Trained user vector[dict]
    :param item_vec: Trained item vector[dict]
    :param user_id: Target user id[str]
    :param top_n: Top n number to be recommended[int]
    :return: A list with n items which consisted of a item id and score tuple[list]
    """
    if user_id not in user_vec:
        return []
    record = {}
    rec_list = []
    user_vector = user_vec[user_id]
    for item_id in item_vec:
        item_vector = item_vec[item_id]
        distance = LFM.model_predict(user_vector, item_vector)
        record[item_id] = distance
    for item_score in sorted(record.items(), key=lambda element: element[1], reverse=True)[:top_n]:
        item_id = item_score[0]
        score = round(item_score[1], 3)
        rec_list.append((item_id, score))
    return rec_list


def rec_analysis(target_user, **params):
    """
    Analysis recommended item information
    :param target_user: Target user id[str]
    :return: Nothing
    """
    item_info = get_movie_info(params["movie_info"])
    user_vec, item_vec = get_vectors(params["uv_file"], params["iv_file"])
    rating_data = get_train_data(input_path=params["input_file"], threshold=params["threshold"], sep=",")
    for data in rating_data:
        user_id, item_id, label = data
        if user_id == target_user and label == 1:
            print(item_info[item_id])
    print("\nrecommend list")
    recommend_list = rec_top_n(user_vec, item_vec, target_user, params["top_n"])
    for rec_item in recommend_list:
        print(item_info[rec_item[0]], rec_item[1])


if __name__ == '__main__':
    # Run parameters
    params_dict = {"latent_factor": 50, "alpha": 0.01, "learning_rate": 0.01, "step": 10, "threshold": 4,
                   "input_file": "../data/movieLens25m_201912/ratings.csv", "uv_file": "vecData/user_vec.txt",
                   "iv_file": "vecData/item_vec.txt", "movie_info": "../data/movieLens25m_201912/movies.csv", "top_n": 10}
    # Train lfm model
    train_model(**params_dict)
    # Recommend top n
    rec_analysis("999999", **params_dict)
