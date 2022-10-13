# Author: Hayden Lao
# Script Name: model_item2vec_gensim
# Created Date: Mar 8th 2020
# Description: Item2Vec model built by gensim for movieLens recommendation

from gensim.models import Word2Vec


class Item2VecGensim:
    def __init__(self, **params):
        self.act_seq_file = params["act_seq_file"]  # action sequence file
        self.act_seq_separator = params["act_seq_separator"]  # action sequence separator
        self.gensim_model_file = params["gensim_model_file"]  # gensim output word2vec model file
        self.gensim_vectors_file = params["gensim_vectors_file"]  # gensim output vectors file
        self.learning_rate = params["learning_rate"]  # learning rate
        self.emb_size = params["emb_size"]  # vector dimension
        self.window = params["window"]  # slide window
        self.min_count = params["min_count"]  # minimum action number
        self.network_algo = params["network_algo"]  # network algorithm: 0 for CBOW, 1 for skip-gram
        self.speed_algo = params["speed_algo"]  # speed algorithm: 0 for negative sampling, 1 for hierarchical softmax
        self.negative_number = params["negative_number"]  # negative sampling number

    def __iter__(self):
        for line in open(self.act_seq_file):
            yield line.split(self.act_seq_separator)

    def init_model(self, bow):
        """
        Initialize train model
        :param bow: Bag of words, a iterable object[object/list]
        :return: Gensim word2vec model[object]
        """
        model = Word2Vec(bow, alpha=self.learning_rate, size=self.emb_size, window=self.window,
                         min_count=self.min_count, sg=self.network_algo, hs=self.speed_algo,
                         negative=self.negative_number)
        return model

    def save_vectors(self, model):
        """
        Save word2vec vectors into file in gensim format
        :param model: Input gensim word2vec model[object]
        :return: Nothing return
        """
        model.wv.save_word2vec_format(self.gensim_vectors_file)

    def save_model(self, model):
        """
        Save word2vec model into file
        :param model: Gensim word2vec model
        :return: Nothing return
        """
        model.save(self.gensim_model_file)

    def topn_similar(self, model, item_id, topn):
        """
        Get top n most similar item by cosine similarity using gensim model
        :param model: Input gensim word2vec model[object]
        :param item_id:
        :param topn:
        :return:
        """
        recom_list = model.most_similar(item_id, topn=topn)
        return recom_list
