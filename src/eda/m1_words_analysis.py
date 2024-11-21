# --------------------------------------------------------------
# -- EDA M1 :: Words Analysis Module ---------------------------
# --------------------------------------------------------------

import re
import itertools
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from networkx.algorithms.community import greedy_modularity_communities, k_clique_communities

# --------------------------------------------------------------
# -- Text Cleaning --------------------------------------------
# --------------------------------------------------------------
class TextCleaner:
    """Cleans and processes text data."""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.custom_exclusions = { 'also','vol','trimsize','et', 'al', 'cid', 'fig', 'k', 'v', 'b', 'n', 'p', 'f', 'h', 'e', 'g', 'ieee'}

    def clean(self, text):
        """Clean text by removing stopwords, single letters, numbers, and custom exclusions."""
        if not isinstance(text, str):
            return []
        text = re.sub(r'[^a-z\s]', '', text.lower())  # Remove non-alphabetic characters
        tokens = word_tokenize(text)
        return [
            word for word in tokens
            if word not in self.stop_words and
               word not in self.custom_exclusions and
               len(word) > 1
        ]

# --------------------------------------------------------------
# -- Word Processor -------------------------------------------
# --------------------------------------------------------------
class Processor:
    """Processes word data for analysis."""
    
    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.cleaner = TextCleaner()

    def get_most_frequent_words(self, n=20):
        """Get the most frequent words in the plain text."""
        all_words = self._extract_words_from_plain_text()
        word_counts = Counter(all_words)
        return pd.DataFrame(word_counts.most_common(n), columns=["Word", "Count"])

    def get_top_keywords(self, n=10):
        """Get the top keywords from titles, keywords, and descriptions."""
        all_keywords = []
        for entry in self.processed_data:
            combined_text = " ".join(
                entry['bibliographic_metadata'].get(field, '')
                for field in ['keywords', 'title', 'description']
            )
            all_keywords.extend(self.cleaner.clean(combined_text))
        keyword_counts = Counter(all_keywords)
        return pd.DataFrame(keyword_counts.most_common(n), columns=["Keyword", "Count"])

    def get_word_trends(self, words_to_track):
        """Get trends of specific words over time."""
        trends = []
        for entry in self.processed_data:
            year = self._parse_year(entry['bibliographic_metadata'].get('year'))
            if not year:
                continue
            
            cleaned_words = Counter(self.cleaner.clean(entry.get('plain_text', '')))
            
            for word in words_to_track:
                trends.append({
                    'Year': year,
                    'Word': word,
                    'Count': cleaned_words.get(word, 0)
                })
        
        return pd.DataFrame(trends).groupby(['Year', 'Word'], as_index=False).sum()

    def get_cooccurrence_matrix(self, top_words):
        """Generate a co-occurrence matrix for the top words."""
        cooccurrence_counts = defaultdict(int)
        for entry in self.processed_data:
            filtered_words = [
                word for word in self.cleaner.clean(entry.get('plain_text', ''))
                if word in top_words
            ]
            for pair in itertools.combinations(set(filtered_words), 2):
                cooccurrence_counts[tuple(sorted(pair))] += 1
        return pd.DataFrame(
            [{'Word1': w1, 'Word2': w2, 'Count': count} for (w1, w2), count in cooccurrence_counts.items()]
        )

    def _extract_words_from_plain_text(self):
        """Extract cleaned words from all plain text in the dataset."""
        return [
            word for entry in self.processed_data
            for word in self.cleaner.clean(entry.get('plain_text', ''))
        ]

    @staticmethod
    def _parse_year(year):
        """Convert year to integer if valid, otherwise return None."""
        try:
            year_int = int(year)
            return year_int if year_int > 0 else None
        except (ValueError, TypeError):
            return None

# --------------------------------------------------------------
# -- Visualization --------------------------------------------
# --------------------------------------------------------------
class Visualizer:
    """Handles visualization of word analysis results."""

    @staticmethod
    def plot_bar_chart(df, title, filename, x_col="Word", y_col="Count"):
        """Generic method to plot bar charts."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[x_col], df[y_col], color="#87CEEB")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(f"../output/m1_words/{filename}.svg", format="svg")
        fig.savefig(f"../output/m1_words/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_wordcloud(word_freq, title, filename):
        """Generate and save a word cloud."""
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(f"../output/m1_words/{filename}.svg", format="svg")
        fig.savefig(f"../output/m1_words/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_line_chart(df, title, filename, x_col="Year", y_col="Count", hue_col="Word"):
        """
        Plot a line chart for trends over time, with each line representing a different term.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot each term as a separate line
        for term in df[hue_col].unique():
            term_data = df[df[hue_col] == term]
            ax.plot(term_data[x_col], term_data[y_col], marker='o', label=term)
        
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.legend(title=hue_col, fontsize=10, loc="upper left")
        ax.grid(True, linestyle="--", linewidth=0.5)
        
        plt.tight_layout()
        fig.savefig(f"../output/m1_words/{filename}.svg", format="svg")
        fig.savefig(f"../output/m1_words/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_cooccurrence_network(cooccurrence_df, filename="cooccurrence_network"):
        """
        Plot a co-occurrence network using NetworkX.
        """
        G = nx.Graph()

        # Add edges with weights
        for _, row in cooccurrence_df.iterrows():
            G.add_edge(row['Word1'], row['Word2'], weight=row['Count'])

        # Generate layout and plot the network
        pos = nx.spring_layout(G, seed=42)  # Position nodes using a spring layout
        plt.figure(figsize=(12, 12))

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")

        # Draw edges with width based on weight
        edges = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges.keys(),
            width=[v * 0.1 for v in edges.values()]
        )

        # Add labels to nodes
        nx.draw_networkx_labels(G, pos, font_size=10, font_color="black")

        # Add title and save the plot
        plt.title("Word Co-Occurrence Network", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"../output/m1_words/{filename}.svg", format="svg")
        plt.savefig(f"../output/m1_words/{filename}.png", format="png")
        plt.show()


# --------------------------------------------------------------
# -- Reporting -----------------------------------------------
# --------------------------------------------------------------
class Reporter:
    """Handles reporting and saving analysis results."""

    @staticmethod
    def save_to_csv(df, filename):
        """Save a DataFrame to a CSV file."""
        if not df.empty:
            df.to_csv(f"../output/m1_words/{filename}.csv", index=False)
