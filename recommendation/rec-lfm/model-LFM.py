# Author: Hayden Lao
# Script Name: model-lfm
# Created Date: Sep 5th 2019
# Description: Latent factor model for movieLens learnTFData recommendation

import numpy as np
from recommendation.learnTFUtils.ReadDataUtils import get_train_data
from recommendation.learnTFUtils.ReadDataUtils import get_movie_info_text


def lfm_train(train_data, latent_factor, alpha, learning_rate, step):
    """
    LFM training model
    :param train_data: Input constructed train data[list]
    :param latent_factor: Latent factor number[int]
    :param alpha: Regularization parameter[float]
    :param learning_rate: Learning rate[float]
    :param step: Training step number[int]
    :return: Tuple contains user vector dictionary and item vector dictionary[tuple]
    """
    user_vec, item_vec = {}, {}
    for step_index in range(step):
        for data in train_data:
            userId, itemId, label = data
            if userId not in user_vec:
                user_vec[userId] = init_model(latent_factor)
            if itemId not in item_vec:
                item_vec[itemId] = init_model(latent_factor)
            delta = label - model_predict(user_vec[userId], item_vec[itemId])
            for i in range(latent_factor):
                user_vec[userId][i] += learning_rate * (delta * item_vec[itemId][i] - alpha * user_vec[userId][i])
                item_vec[itemId][i] += learning_rate * (delta * user_vec[userId][i] - alpha * item_vec[itemId][i])
        # Slower down learning as step goes
        learning_rate *= 0.9
    return user_vec, item_vec


def init_model(latent_factor):
    """
    Initialize latent factor preferences
    :param latent_factor: Number of latent factor[int]
    :return: An array with initialized preference range from 0 to 1[array]
    """
    dist = np.random.rand(latent_factor)
    return dist


def model_predict(user_vec, item_vec):
    """
    Calculate cosine distance between user and item
    :param user_vec: User preference for latent factors[array]
    :param item_vec: Item relevance for latent factors[array]
    :return: Calculated cosine distance[float]
    """
    user_vec_mod = np.linalg.norm(user_vec)
    item_vec_mod = np.linalg.norm(item_vec)
    cosine_distance = np.dot(user_vec, item_vec) / (user_vec_mod * item_vec_mod)
    return cosine_distance


def model_train_process():
    """
    Function to train LFM model
    :return: Nothing
    """
    train_data = get_train_data(input="../learnTFData/ratings.csv", threshold=4)
    result = lfm_train(train_data=train_data, latent_factor=50, alpha=0.01, learning_rate=0.1, step=50)
    user_vec, item_vec = result
    print(user_vec["1"])
    print(item_vec["2554"])


def rec_top_n(user_vec, item_vec, userId, top_n=10):
    """
    Return top n recommendation list
    :param user_vec: Trained user vector[dict]
    :param item_vec: Trained item vector[dict]
    :param userId: Target user id[str]
    :param top_n: Top n number to be recommended[int]
    :return: A list with n items which consisted of a item id and score tuple[list]
    """
    if userId not in user_vec:
        return []
    record = {}
    rec_list = []
    user_vector = user_vec[userId]
    for itemId in item_vec:
        item_vector = item_vec[itemId]
        distance = model_predict(user_vector, item_vector)
        record[itemId] = distance
    for item_score in sorted(record.items(), key=lambda element: element[1], reverse=True)[:top_n]:
        itemId = item_score[0]
        score = round(item_score[1], 3)
        rec_list.append((itemId, score))
    return rec_list


def rec_analysis(train_data, target_user, rec_list):
    """
    Analysis recommended item information
    :param train_data: A list contains tuples with users movies and labels[list]
    :param target_user: Target user id[str]
    :param rec_list: Recommended list[list]
    :return: Nothing
    """
    item_info = get_movie_info_text("../data/movies.csv")
    for data in train_data:
        userId, itemId, label = data
        if userId == target_user and label == 1:
            print(item_info[itemId])
    print("rec_list")
    for rec_item in rec_list:
        print(item_info[rec_item[0]])


if __name__ == '__main__':
    model_train_process()
