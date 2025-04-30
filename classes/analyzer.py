import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import os
from matplotlib.table import Table

class Analyzer:
    GOLD = '#FFD700'
    BLACK = '#000000'
    
    def __init__(self, tfidf_model, lsi_model, newsgroups):
        self.tfidf_model = tfidf_model
        self.lsi_model = lsi_model
        self.newsgroups = newsgroups
        self.output_dirs = {
            'comparisons': "results/search_comparisons",
            'lsi_results': "results/lsi_search_results"
        }
        for dir in self.output_dirs.values():
            os.makedirs(dir, exist_ok=True)
    
    def plot_comparison(self, query, lex_scores, lsi_scores, lex_time, lsi_time):
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        x = np.arange(3)  # Top 3 results
        width = 0.35
        plt.bar(x - width/2, lex_scores[:3], width, label='TF-IDF', color=self.GOLD)
        plt.bar(x + width/2, lsi_scores[:3], width, label='LSI', color=self.BLACK)
        plt.xlabel('Top Results')
        plt.ylabel('Cosine Similarity')
        plt.title(f'Top Result Scores')
        plt.xticks(x, ['1st', '2nd', '3rd'])
        plt.legend()
        
        # Time comparison
        plt.subplot(1, 2, 2)
        plt.bar(['Lexical', 'LSI'], [lex_time*1000, lsi_time*1000], color=[self.GOLD, self.BLACK])
        plt.ylabel('Time (milliseconds)')
        plt.gca().yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f')) 

        plt.suptitle(f'Search Method Comparison: "{query}"', y=1.05)
        plt.tight_layout()
        safe_query = "".join(c for c in query if c.isalnum() or c in " _-")
        plt.savefig(f"{self.output_dirs['comparisons']}/{safe_query}.png", dpi=120, bbox_inches='tight')
        plt.close()
    
    def analyze_query(self, query):
        lex_scores, lex_time = self.tfidf_model.search(query)
        lsi_scores, lsi_time = self.lsi_model.search(query)
        
        lex_top = np.argsort(lex_scores)[-3:][::-1]
        lsi_top = np.argsort(lsi_scores)[-3:][::-1]
       
        self.plot_comparison(
            query,
            lex_scores[lex_top],
            lsi_scores[lsi_top],
            lex_time,
            lsi_time
        )
    
    def plot_category_distribution(self):
        """Plot the document category distribution"""
        categories = self.newsgroups.target_names
        counts = np.bincount(self.newsgroups.target)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
        bars = plt.barh(range(len(categories)), counts, color=colors)
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 20, i, f'{counts[i]:,}', 
                    va='center', fontsize=8)
        
        plt.yticks(range(len(categories)), categories, fontsize=9)
        plt.title('20 Newsgroups - Document Category Distribution', pad=20)
        plt.xlabel('Number of Documents')
        plt.xlim(0, max(counts) * 1.15)
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        
        for spine in ['top', 'right']:
            plt.gca().spines[spine].set_visible(False)
        
        plt.savefig('results/newsgroups_distribution.png', 
                   dpi=120, 
                   bbox_inches='tight',
                   facecolor='white')
        plt.close()
    
    def generate_natural_queries(self):
        print("Generate 50 natural language search queries")
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
    
    def analyze_lsi_queries(self):
        """Analyze all LSI queries and generate visualizations"""
        queries = self.generate_natural_queries()
        results = []
        
        for query in queries:
            score, top_idx = self.lsi_search(query)
            results.append((query, score, self.newsgroups.target[top_idx]))
        
        self.create_score_table_image(results)
        self.plot_score_distribution(results)
    
    def lsi_search(self, query):
        """Perform LSI search and return top result"""
        query_vec = self.lsi_model.tfidf.transform([query])
        query_lsi = self.lsi_model.lsi.transform(query_vec)
        similarities = cosine_similarity(query_lsi, self.lsi_model.X).flatten()
        top_idx = np.argmax(similarities)
        return similarities[top_idx], top_idx
    
    def create_score_table_image(self, results):
        """Create a PNG image of the query-score table with full queries"""
        fig_height = max(6, len(results) * 0.4)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        ax.axis('off')
        
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
        plt.savefig(f"{self.output_dirs['lsi_results']}/query_score_table.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_score_distribution(self, results):
        """Plot histogram of all query scores"""
        plt.figure(figsize=(10, 6))
        scores = [r[1] for r in results]
        plt.hist(scores, bins=15, color='Black', edgecolor='white')
        plt.title('Distribution of LSI Search Scores (50 Queries)')
        plt.xlabel('Cosine Similarity Score')
        plt.ylabel('Number of Queries')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f"{self.output_dirs['lsi_results']}/score_distribution.png", dpi=120, bbox_inches='tight')
        plt.close()