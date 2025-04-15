import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
from matplotlib.table import Table

class TFIDF_Model:
    def __init__(self, data):
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            min_df=5
        )
        self.X = self.tfidf.fit_transform(data)

    def search(self, query):
        start = time.time()
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.X).flatten()
        return similarities, time.time() - start
