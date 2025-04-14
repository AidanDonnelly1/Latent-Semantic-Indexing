import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_20newsgroups

def plot_category_distribution():
    # Load data
    newsgroups = fetch_20newsgroups(subset='all')
    categories = newsgroups.target_names
    counts = np.bincount(newsgroups.target)  # Actual document counts per category
    
    # Create figure
    plt.figure(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    bars = plt.barh(range(len(categories)), counts, color=colors)
    
    # Add count labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 20, i, f'{counts[i]:,}', 
                va='center', fontsize=8)
    
    # Format/Lables
    plt.yticks(range(len(categories)), categories, fontsize=9)
    plt.title('20 Newsgroups - Document Category Distribution', pad=20)
    plt.xlabel('Number of Documents')
    plt.xlim(0, max(counts) * 1.15)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Remove spines
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)
    
    # Save PNG
    plt.savefig('newsgroups_distribution.png', 
               dpi=120, 
               bbox_inches='tight',
               facecolor='white')
    print("Saved accurate distribution chart to 'newsgroups_distribution.png'")

plot_category_distribution()