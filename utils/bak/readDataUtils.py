# Author: Hayden Lao
# Script Name: ReadDataUtils
# Created Date: Sep 5th 2019
# Description: Some useful data reading methods

import os


def get_movie_info(input_path, sep=","):
    """
    Extract movie basic information into dictionary
    :param input_path: Movie basic information file path[str]
    :param sep: Separation delimiter[str]
    :return: Dictionary with key as movie id and value as list of title and genres[dict]
    """
    if not os.path.exists(input_path):
        print("No such file")
        return {}
    with open(input_path, encoding='UTF-8') as file:
        line_num = 0
        movie_dict = {}
        for line in file:
            if line_num == 0:
                line_num += 1
                continue
            movie_info = line.strip().split(sep)
            if len(movie_info) < 3:
                continue
            elif len(movie_info) == 3:
                movie_id = movie_info[0]
                title = movie_info[1]
                genres = movie_info[2]
            elif len(movie_info) > 3:
                movie_id = movie_info[0]
                title = ",".join(movie_info[1:-1])
                genres = movie_info[-1]
            line_num += 1
            movie_dict[movie_id] = [title, genres]
    return movie_dict


if __name__ == '__main__':
    pass
    # train_data = get_train_data(input_path="../data/ratings.csv", threshold=4, sep=",")
    # print(train_data[:20])
