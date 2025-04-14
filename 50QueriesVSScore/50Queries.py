import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import os
from matplotlib.table import Table

OUTPUT_DIR = "lsi_search_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    """Load and preprocess the 20 Newsgroups dataset"""
    return fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        shuffle=True,
        random_state=42
    )

def build_lsi_model(newsgroups):
    """Build LSI model with preprocessing"""
    tfidf = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        min_df=5
    )
    X_tfidf = tfidf.fit_transform(newsgroups.data)
    
    lsi = TruncatedSVD(n_components=100, random_state=42)
    X_lsi = lsi.fit_transform(X_tfidf)
    
    return tfidf, lsi, X_lsi, newsgroups.target

def generate_natural_queries():
    """Generate 50 natural language search queries"""
    return [
        "How does NASA prepare for space shuttle launches?",
        "Common driver issues in Windows operating systems",
        "Current standings in the NHL hockey playoffs",
        "Troubleshooting car engine problems in cold weather",
        "Core beliefs of Christianity compared to other religions",
        "Performance benchmarks for latest graphics cards",
        "Recent breakthroughs in medical cancer research",
        "Accuracy of political election polls in swing states",
        "Best universities for computer science education",
        "Career statistics for legendary baseball players",
        "Life aboard the International Space Station",
        "Comparing Linux distributions for beginners",
        "Analyzing stock market trends during recessions",
        "Philosophical debates about religious experiences",
        "Safety ratings for popular family automobiles",
        "Efficiency improvements in solar energy technology",
        "Challenges in Middle East peace negotiations",
        "Recent developments in gun control legislation",
        "Modern applications of cryptography algorithms",
        "How music synthesizers create different sounds",
        "New discoveries from Hubble telescope images",
        "Protecting your computer from virus attacks",
        "Predictions for the NBA basketball championship",
        "Battery technology in electric vehicles",
        "Different interpretations of the Bible",
        "Debates between atheism and organized religion",
        "Technical specifications of military aircraft",
        "Organic gardening techniques for urban areas",
        "Maintenance tips for motorcycle owners",
        "Philosophical arguments for atheism",
        "Ethical concerns about artificial intelligence",
        "Future missions in NASA's space program",
        "Solving common Windows update problems",
        "Analysis of hockey team power rankings",
        "Improving car fuel efficiency in city driving",
        "Theological differences among Christian denominations",
        "Advances in computer graphics technology",
        "Promising new directions in cancer research",
        "Fundraising strategies in presidential campaigns",
        "Learning resources for computer programming",
        "Controversies surrounding Baseball Hall of Fame",
        "Private companies involved in space exploration",
        "Benefits of open source software development",
        "Safe investment strategies during inflation",
        "Core teachings of Buddhism for daily life",
        "Statistics on car accidents caused by texting",
        "Government incentives for renewable energy",
        "Historical roots of Middle East conflicts",
        "Legal interpretations of Second Amendment rights",
        "How data encryption protects online privacy"
    ]

def lsi_search(tfidf, lsi, X_lsi, query):
    """Perform LSI search and return top result"""
    query_vec = tfidf.transform([query])
    query_lsi = lsi.transform(query_vec)
    similarities = cosine_similarity(query_lsi, X_lsi).flatten()
    top_idx = np.argmax(similarities)
    return similarities[top_idx], top_idx

def analyze_queries(tfidf, lsi, X_lsi, target, queries):
    """Analyze all queries and return results"""
    results = []
    for query in queries:
        score, top_idx = lsi_search(tfidf, lsi, X_lsi, query)
        results.append((query, score, target[top_idx]))
    return results

def create_score_table_image(results):
    """Create a PNG image of the query-score table with full queries"""
    fig_height = max(6, len(results) * 0.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis('off')
    
    # Sort results by score (descending)
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    table = ax.table(
        cellText=[[query, f"{score:.4f}"] for query, score, _ in sorted_results],
        colLabels=["Search Query", "Score"],
        loc='center',
        cellLoc='left',
        colColours=['black', 'black'],
        colWidths=[0.75, 0.25]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 1.75)

    for key, cell in table.get_celld().items():
        cell.set_text_props(wrap=True)
        if key[0] == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('black')
        else:
            cell.set_facecolor('white')
    
    plt.title("LSI Search Results - Query vs Score (TRAIN Data)", y=1.02, fontsize=24)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/query_score_table.png", dpi=150, bbox_inches='tight')
    plt.close()

def plot_score_distribution(results):
    """Plot histogram of all query scores"""
    plt.figure(figsize=(10, 6))
    scores = [r[1] for r in results]
    plt.hist(scores, bins=15, color='Black', edgecolor='white')
    plt.title('Distribution of LSI Search Scores (50 Queries)')
    plt.xlabel('Cosine Similarity Score')
    plt.ylabel('Number of Queries')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{OUTPUT_DIR}/score_distribution.png", dpi=120, bbox_inches='tight')
    plt.close()


def main():
    print("Loading data and building LSI model...")
    newsgroups = load_data()
    tfidf, lsi, X_lsi, target = build_lsi_model(newsgroups)
    
    print("Generating 50 natural language queries...")
    queries = generate_natural_queries()
    
    print("Running LSI searches...")
    results = analyze_queries(tfidf, lsi, X_lsi, target, queries)
    
    print("Creating visualizations...")
    create_score_table_image(results)
    plot_score_distribution(results)
    
    print(f"\nResults saved to '{OUTPUT_DIR}/':")
    print("- query_score_table.png (complete results with full queries)")
    print("- score_distribution.png")

if __name__ == "__main__":
    main()