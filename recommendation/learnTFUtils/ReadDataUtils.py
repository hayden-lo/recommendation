# Author: Hayden Lao
# Script Name: ReadDataUtils
# Created Date: Sep 5th 2019
# Description: Some useful data reading methods

import os


def get_movie_info_text(input, sep=","):
    """
    Extract movie basic information into dictionary
    :param input: Movie basic information file path[str]
    :param sep: Separation delimiter[str]
    :return: Dictionary with key as movie id and value as list of title and genres[dict]
    """
    if not os.path.exists(input):
        print("No such file")
        return {}
    with open(input, encoding='UTF-8') as file:
        linenum = 0
        movieDict = {}
        for line in file:
            if linenum == 0:
                linenum += 1
                continue
            movieInfo = line.strip().split(sep)
            if len(movieInfo) < 3:
                continue
            elif len(movieInfo) == 3:
                movieId = movieInfo[0]
                title = movieInfo[1]
                genres = movieInfo[2]
            elif len(movieInfo) > 3:
                movieId = movieInfo[0]
                title = ",".join(movieInfo[1:-1])
                genres = movieInfo[-1]
            linenum += 1
            movieDict[movieId] = [title, genres]
    return movieDict


def get_movie_avgRating_txt(input, sep=","):
    """
    Extract average rating for each movie
    :param input: Movie Rating file path[str]
    :param sep: Separation delimiter[str]
    :return: Dictionary with key as movie id and value as average rating[dict]
    """
    if not os.path.exists(input):
        print("No such file")
        return {}
    with open(input, encoding='UTF-8') as file:
        linenum = 0
        ratingDict = {}
        for line in file:
            if linenum == 0:
                linenum += 1
                continue
            movieRating = line.strip().split(sep)
            if len(movieRating) == 4:
                movieId = movieRating[1]
                rating = movieRating[2]
            else:
                continue
            if movieId not in ratingDict.keys():
                ratingDict[movieId] = [1, float(rating)]
            else:
                ratingDict[movieId][0] += 1
                ratingDict[movieId][1] += float(rating)
            linenum += 1
    for movie in ratingDict.keys():
        ratingDict[movie] = round(ratingDict[movie][1] / ratingDict[movie][0], 2)
    return ratingDict


def get_train_data(input, threshold, sep=","):
    """
    Make train data based on rating behavior
    :param input: Movie rating file path[str]
    :param threshold: Thresold for defining positive data[int/float]
    :param sep: Separation delimiter[str]
    :return: A list contains tuples with users movies and labels[list]
    """
    if not os.path.exists(input):
        print("No such file")
        return {}
    train_data = []
    ratingDict = get_movie_avgRating_txt(input)
    with open(input, encoding="UTF-8") as file:
        posDict, negDict = {}, {}
        linenum = 0
        for line in file:
            if linenum == 0:
                linenum += 1
                continue
            movieRating = line.strip().split(sep)
            if len(movieRating) == 4:
                userId = movieRating[0]
                movieId = movieRating[1]
                rating = movieRating[2]
            else:
                continue
            if float(rating) >= threshold:
                if userId not in posDict.keys():
                    posDict[userId] = [(movieId, 1)]
                else:
                    posDict[userId].append((movieId, 1))
            else:
                if userId not in negDict.keys():
                    negDict[userId] = [(movieId, ratingDict.get(movieId, 0))]
                else:
                    negDict[userId].append((movieId, ratingDict.get(movieId, 0)))
    for user in posDict.keys():
        data_num = min(len(posDict[user]), len(negDict.get(user, [])))
        if data_num > 0:
            # Construct label proportion to 1:1
            negDict[user] = sorted(negDict[user], key=lambda element: element[1], reverse=True)[:data_num]
            train_data += [(user, action[0], action[1]) for action in posDict[user]]
            train_data += [(user, action[0], 0) for action in negDict[user]]
        else:
            continue
    return train_data


if __name__ == '__main__':
    train_data = get_train_data(input="../learnTFData/ratings.csv", threshold=4, sep=",")
    print(train_data[:20])
