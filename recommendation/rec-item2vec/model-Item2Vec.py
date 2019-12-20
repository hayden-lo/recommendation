# Author: Hayden Lao
# Script Name: model-item2vec
# Created Date: Nov 11th 2019
# Description: Item2Vec for movieLens learnTFData recommendation

import os

def produce_train_data(input,output,sep=","):
    """
    Produce training data for Word2Vec
    :param input: Input data path[string]
    :param output: Output path[string]
    :return:
    """
    if os.path.exists(input):
        return
    with open(input, encoding='UTF-8') as file:
        linenum=0
        for line in file:
            if linenum==0:
                linenum+=1
                continue
            item=line.strip().split(sep)
