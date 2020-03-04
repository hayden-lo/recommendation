# Author: Hayden Lao
# Script Name: model_item2vec
# Created Date: Nov 11th 2019
# Description: Item2Vec for movieLens recData recommendation

import os


def produce_train_data(input_path, output_path, sep=",", threshold=4.0):
    """
    Produce training data for Word2Vec
    :param input_path: Input_path data path[string]
    :param output_path: Output path path[string]
    :param sep: Separator of the input file data[str]
    :param threshold: Rating threshold to be written in the output file[float]
    :return: Produce train data txt in output path, nothing return
    """
    if not os.path.exists(input_path):
        print("No such file")
        return
    record = {}
    with open(input_path, encoding='UTF-8') as input_file:
        line_num = 0
        for line in input_file:
            if line_num == 0:
                line_num += 1
                continue
            item = line.strip().split(sep)
            if len(item) < 4:
                continue
            user_id, item_id, rating = item[0], item[1], float(item[2])
            if rating < threshold:
                continue
            if user_id not in record:
                record[user_id] = []
            record[user_id].append(item_id)
    with open(output_path, "w") as output_file:
        for user_id in record:
            output_file.write(" ".join(record[user_id]) + "\n")


if __name__ == "__main__":
    produce_train_data("../recData/ratings.csv", "../recData/test.txt", ",", 4)
