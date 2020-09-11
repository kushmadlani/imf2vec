
import pandas as pd 
import numpy as np
import pickle

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


class ItemSequences:
    '''
    ItemSequences
    '''
    def __init__(self, df_column):
        df_column = df_column.copy()
        self.sequences = df_column.values.tolist()

    def __iter__(self):
        for sequence in self.sequences:
            yield sequence.split(',')


class Item2Vec:
    def __init__(self, df, column_name, window_size, embedding_size):
        self.df = df[column_name]
        self.window_size = window_size
        self.embedding_size = embedding_size    
        self.embedding_matrix, self.items = self.get_embedding()

    def get_embedding(self):
        """Generate word2vec style embeddings trained on sequences of events using Skip-gram loss"""
        sentences = ItemSequences(self.df)
        model = Word2Vec(sentences, window=self.window_size, min_count=5, sg=1, size=self.embedding_size)
        items = list(map(int,model.wv.index2word))
        return model.wv.vectors, items

    def similarity(self, train_items=None):
        """Create similarity matrix and nearest neighbours for items"""
        if train_items:
            self.valid_items = list(set(train_items) & set(self.items))
            self.check_valid = [True if i in self.valid_items else False for i in train_items]
            valid_keys = [self.items.index(i) for i in self.valid_items]
            valid_embedding_matrix = self.embedding_matrix[valid_keys]
            self.reverse_lookup = {i:train_items.index(j) for i,j in enumerate(self.valid_items)}
        else:
            valid_embedding_matrix = self.embedding_matrix

        self.S = cosine_similarity(valid_embedding_matrix)
        self.top_k = np.argsort(-self.S,axis=1)[:,1:]
