import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
from matplotlib.table import Table

class LSI_Model:
    def __init__(self, tfidf_model):
        self.lsi = TruncatedSVD(n_components=100, random_state=42)
        self.X = self.lsi.fit_transform(tfidf_model.X)
        self.tfidf = tfidf_model.tfidf
    
    def search(self, query):
        start = time.time()
        query_vec = self.tfidf.transform([query])
        query_lsi = self.lsi.transform(query_vec)
        similarities = cosine_similarity(query_lsi, self.X).flatten()
        return similarities, time.time() - start