# Author: Hayden Lao
# Script Name: iter_consturc
# Created Date: Apr 9th 2020
# Description: Iterator constructor for item2vec

import os

class IterConstructor:
    def __init__(self, dirname):
        self.dirname = dirname  # input file path

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
