# Author: Hayden Lao
# Script Name: get_train_data
# Created Date: Sep 5th 2019
# Description: Construct train data for LFM model

import os


def get_movie_avg_rating(input_path, sep=","):
    """
    Extract average rating for each movie
    :param input_path: Movie Rating file path[str]
    :param sep: Separation delimiter[str]
    :return: Dictionary with key as movie id and value as average rating[dict]
    """
    if not os.path.exists(input_path):
        print("No such file")
        return {}
    with open(input_path, encoding='UTF-8') as file:
        line_num = 0
        rating_dict = {}
        for line in file:
            if line_num == 0:
                line_num += 1
                continue
            movie_rating = line.strip().split(sep)
            if len(movie_rating) == 4:
                movie_id = movie_rating[1]
                rating = movie_rating[2]
            else:
                continue
            if movie_id not in rating_dict.keys():
                rating_dict[movie_id] = [1, float(rating)]
            else:
                rating_dict[movie_id][0] += 1
                rating_dict[movie_id][1] += float(rating)
            line_num += 1
    for movie in rating_dict.keys():
        rating_dict[movie] = round(rating_dict[movie][1] / rating_dict[movie][0], 2)
    return rating_dict


def get_train_data(input_path, threshold, sep=","):
    """
    Make train data based on rating behavior
    :param input_path: Movie rating file path[str]
    :param threshold: Threshold for defining positive data[int/float]
    :param sep: Separation delimiter[str]
    :return: A list contains tuples with users movies and labels[list]
    """
    if not os.path.exists(input_path):
        print("No such file")
        return {}
    train_list = []
    rating_dict = get_movie_avg_rating(input_path)
    with open(input_path, encoding="UTF-8") as file:
        pos_dict, neg_dict = {}, {}
        line_num = 0
        for line in file:
            if line_num == 0:
                line_num += 1
                continue
            movie_rating = line.strip().split(sep)
            if len(movie_rating) == 4:
                user_id = movie_rating[0]
                movie_id = movie_rating[1]
                rating = movie_rating[2]
            else:
                continue
            if float(rating) >= threshold:
                if user_id not in pos_dict.keys():
                    pos_dict[user_id] = [(movie_id, 1)]
                else:
                    pos_dict[user_id].append((movie_id, 1))
            else:
                if user_id not in neg_dict.keys():
                    neg_dict[user_id] = [(movie_id, rating_dict.get(movie_id, 0))]
                else:
                    neg_dict[user_id].append((movie_id, rating_dict.get(movie_id, 0)))
    for user in pos_dict.keys():
        data_num = min(len(pos_dict[user]), len(neg_dict.get(user, [])))
        if data_num > 0:
            # Construct label proportion to 1:1
            neg_dict[user] = sorted(neg_dict[user], key=lambda element: element[1], reverse=True)[:data_num]
            train_list += [(user, action[0], action[1]) for action in pos_dict[user]]
            train_list += [(user, action[0], 0) for action in neg_dict[user]]
        else:
            continue
    return train_list
