# Author: Hayden Lao
# Script Name: get_train_data
# Created Date: Nov 11th 2019
# Description: Construct train data for Item2vec model

import os


def produce_train_data(input_path, output_path, sep=",", threshold=4.0, max_seq=30):
    """
    Produce training data for Word2Vec
    :param input_path: Input_path data path[string]
    :param output_path: Output path path[string]
    :param sep: Separator of the input file data[str]
    :param threshold: Rating threshold to be written in the output file[float]
    :param max_seq: Maximum sequence length[int]
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
            user_id, item_id, rating, timestamp = item[0], item[1], float(item[2]), float(item[3])
            if rating < threshold:
                continue
            if user_id not in record:
                record[user_id] = []
            record[user_id].append((item_id, timestamp))
    with open(output_path, "w") as output_file:
        for user_id in record:
            record[user_id].sort(key=lambda t: t[1], reverse=True)
            action_seq = [item[0] for item in record[user_id]][:max_seq]
            output_file.write(" ".join(action_seq) + "\n")
