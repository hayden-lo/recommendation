# Author: Hayden Lao
# Script Name: model_lfm
# Created Date: Sep 5th 2019
# Description: Latent factor model for movieLens data recommendation

import os
import numpy as np
from models.lfm.get_train_data import get_train_data


class LFM:
    def __init__(self, **params):
        self.latent_factor = params["latent_factor"]
        self.alpha = params["alpha"]
        self.learning_rate = params["learning_rate"]
        self.step = params["step"]
        self.threshold = params["threshold"]

    def lfm_train(self, train_data):
        """
        LFM training model
        :param train_data: Input constructed train data[list]
        :return: Tuple contains user vector dictionary and item vector dictionary[tuple]
        """
        user_vec, item_vec = {}, {}
        for step_index in range(self.step):
            # progress_bar(step_index + 1, self.step)
            print(step_index)
            num=0
            for data in train_data:
                num+=1
                if num%1000==0:
                    print(step_index,num,len(train_data))
                user_id, item_id, label = data
                if user_id not in user_vec:
                    user_vec[user_id] = self.init_model()
                if item_id not in item_vec:
                    item_vec[item_id] = self.init_model()
                delta = label - self.model_predict(user_vec[user_id], item_vec[item_id])
                for i in range(self.latent_factor):
                    user_vec[user_id][i] += self.learning_rate * (
                            delta * item_vec[item_id][i] - self.alpha * user_vec[user_id][i])
                    item_vec[item_id][i] += self.learning_rate * (
                            delta * user_vec[user_id][i] - self.alpha * item_vec[item_id][i])
            # Slower down learning as step goes
            self.learning_rate *= 0.9
        return user_vec, item_vec

    def init_model(self):
        """
        Initialize latent factor preferences
        :return: An array with initialized preference range from 0 to 1[array]
        """
        dist = np.random.rand(self.latent_factor)
        return dist

    @staticmethod
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

    def model_train_process(self, input_file, user_outfile, item_outfile):
        """
        Function to train LFM model
        :return: Nothing
        """
        train_data = get_train_data(input_file, threshold=self.threshold)
        result = self.lfm_train(train_data=train_data)
        user_vec, item_vec = result
        # Check whether input file exists
        if not os.path.exists(input_file):
            print("No such input file")
            return
        # Create output directory if not exists
        output_dirname = os.path.dirname(user_outfile)
        if not os.path.exists(output_dirname):
            os.mkdir(output_dirname)
        # Write user vector output file
        self.write_lfm_vector(user_outfile, user_vec)
        # Write item vector output file
        self.write_lfm_vector(item_outfile, item_vec)

    def write_lfm_vector(self, output_file, vectors):
        """
        Write lfm vectors into output file
        :param output_file: Output file path[str]
        :param vectors: User or item vectors[dict]
        :return: Write data into file, nothing return
        """
        with open(output_file, "w") as file:
            for v in vectors:
                if len(vectors[v]) != self.latent_factor:
                    continue
                file.write(v + " ")
                file.write(",".join([str(i) for i in vectors[v]]) + "\n")
