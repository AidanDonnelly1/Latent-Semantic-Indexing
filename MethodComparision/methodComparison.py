# search_comparator.py (updated with only bar colors changed)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import time
import os

OUTPUT_DIR = "search_comparisons"
os.makedirs(OUTPUT_DIR, exist_ok=True)

GOLD = '#FFD700'
BLACK = '#000000'

def load_data():
    """Load and preprocess the 20 Newsgroups data"""
    return fetch_20newsgroups(
        subset='all',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )

def build_models(newsgroups):
    """Build both TF-IDF and LSI models"""
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        min_df=5
    )
    X_tfidf = tfidf.fit_transform(newsgroups.data)
    
    lsi = TruncatedSVD(n_components=100, random_state=42)
    X_lsi = lsi.fit_transform(X_tfidf)
    
    return tfidf, X_tfidf, lsi, X_lsi

def lexical_search(tfidf, X_tfidf, query):
    """Pure lexical matching using TF-IDF"""
    start = time.time()
    query_vec = tfidf.transform([query])
    similarities = cosine_similarity(query_vec, X_tfidf).flatten()
    return similarities, time.time() - start

def lsi_search(tfidf, lsi, X_lsi, query):
    """Semantic search using LSI"""
    start = time.time()
    query_vec = tfidf.transform([query])
    query_lsi = lsi.transform(query_vec)
    similarities = cosine_similarity(query_lsi, X_lsi).flatten()
    return similarities, time.time() - start
def plot_comparison(query, lex_scores, lsi_scores, lex_time, lsi_time):
    """Generate and save comparison plot with black and gold bars only"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    x = np.arange(3)  # Top 3 results
    width = 0.35
    plt.bar(x - width/2, lex_scores[:3], width, label='Lexical Matching', color=GOLD)
    plt.bar(x + width/2, lsi_scores[:3], width, label='LSI', color=BLACK)
    plt.xlabel('Top Results')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Top Result Scores')
    plt.xticks(x, ['1st', '2nd', '3rd'])
    plt.legend()
    
    # Time comparison
    plt.subplot(1, 2, 2)
    plt.bar(['Lexical', 'LSI'], [lex_time*1000, lsi_time*1000], color=[GOLD, BLACK])
    plt.ylabel('Time (milliseconds)')
    plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) 

    plt.suptitle(f'Search Method Comparison: "{query}"', y=1.05)
    plt.tight_layout()
    safe_query = "".join(c for c in query if c.isalnum() or c in " _-")
    plt.savefig(f"{OUTPUT_DIR}/{safe_query}.png", dpi=120, bbox_inches='tight')
    plt.close()

def analyze_query(newsgroups, tfidf, X_tfidf, lsi, X_lsi, query):
    lex_scores, lex_time = lexical_search(tfidf, X_tfidf, query)
    lsi_scores, lsi_time = lsi_search(tfidf, lsi, X_lsi, query)
    
    lex_top = np.argsort(lex_scores)[-3:][::-1]
    lsi_top = np.argsort(lsi_scores)[-3:][::-1]
   
    plot_comparison(
        query,
        lex_scores[lex_top],
        lsi_scores[lsi_top],
        lex_time,
        lsi_time
    )

def main():
    print("Loading data and building models...")
    newsgroups = load_data()
    tfidf, X_tfidf, lsi, X_lsi = build_models(newsgroups)
    
    # Test queries
    queries = [
        "space shuttle launch",
        "windows driver issues",
        "hockey playoffs",
        "car engine problems",
        "christian beliefs"
    ]
    
    print("Testing queries...")
    for query in queries:
        analyze_query(newsgroups, tfidf, X_tfidf, lsi, X_lsi, query)
    
    print(f"Comparison plots saved to '{OUTPUT_DIR}/' directory")

if __name__ == "__main__":
    main()