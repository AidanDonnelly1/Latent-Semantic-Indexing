# main.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
from matplotlib.table import Table

from classes.lsi_model import LSI_Model
from classes.tfidf_model import TFIDF_Model
from classes.analyzer import Analyzer

def load_data():
    """Load and preprocess the 20 Newsgroups data"""
    return fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )

def main():
    print("Loading data...")
    newsgroups = load_data()
    
    print("Building TF-IDF model...")
    tfidf_model = TFIDF_Model(newsgroups.data)
    
    print("Building LSI model...")
    lsi_model = LSI_Model(tfidf_model)
    
    print("Creating analyzer...")
    analyzer = Analyzer(tfidf_model, lsi_model, newsgroups)
    
    # Generate all visualizations
    print("Generating category distribution plot...")
    analyzer.plot_category_distribution()
    
    print("Running comparison queries...")
    queries = [
        "space shuttle launch",
        "windows driver issues",
        "hockey playoffs",
        "car engine problems",
        "christian beliefs"
    ]
    for query in queries:
        analyzer.analyze_query(query)
    
    print("Analyzing 50 natural language queries...")
    analyzer.analyze_lsi_queries()
    
    print("\nAll visualizations generated:")
    print("- newsgroups_distribution.png")
    print("- Comparison plots in 'search_comparisons/' directory")
    print("- LSI query analysis in 'lsi_search_results/' directory")

if __name__ == "__main__":
    main()