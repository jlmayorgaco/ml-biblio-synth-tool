import re
import itertools

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from collections import Counter, defaultdict
from networkx.algorithms.community import greedy_modularity_communities

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# --------------------------------------------------------------
# -- Word Processor -------------------------------------------
# --------------------------------------------------------------
class Processor:
    """Processes author data for analysis."""

    def __init__(self, processed_data):
        self.processed_data = processed_data

    @staticmethod
    def parse_year(year):
        """Convert year to integer if valid, otherwise return None."""
        try:
            return int(year) if int(year) > 0 else None
        except (ValueError, TypeError):
            return None

    def _clean_text(self, text):
        """Clean text by removing stopwords, non-alphabetic characters, and tokenizing."""

        # Initialize stopwords (you may adjust the stopwords list)
        stop_words = set(stopwords.words('english'))

        # Check if text is valid
        if not isinstance(text, str):
            return []

        # Convert to lowercase and remove non-alphabetic characters
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)

        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]

        return filtered_tokens
    
    def get_top_sources(self, n=10):
        """Get the top sources (journals or publishers) by document count."""
        source_counts = []

        for entry in self.processed_data:
            source = entry['bibliographic_metadata'].get('publisher', '')
            if source:
                source_counts.append(source.strip().lower())

        # Count occurrences of each source
        source_frequency = Counter(source_counts)

        # Convert to DataFrame and sort
        top_sources = pd.DataFrame(source_frequency.most_common(n), columns=["Source", "Document Count"])
        return top_sources
    
    def calculate_high_impact_proportion(self, n=5):
        """Dynamically calculate the proportion of articles in the top N high-impact journals."""
        source_counts = []

        for entry in self.processed_data:
            source = entry['bibliographic_metadata'].get('publisher', '').strip().lower()
            if source:
                source_counts.append(source)

        # Get the top N sources as the high-impact list
        high_impact_list = [
            source for source, _ in Counter(source_counts).most_common(n)
        ]

        # Calculate total articles and those in high-impact journals
        total_articles = len(self.processed_data)
        high_impact_count = sum(1 for source in source_counts if source in high_impact_list)

        # Create proportions DataFrame
        proportions = {
            "Category": ["High Impact Journals", "Other Journals"],
            "Article Count": [high_impact_count, total_articles - high_impact_count]
        }
        proportions_df = pd.DataFrame(proportions)

        # Return both the proportions DataFrame and the dynamic high-impact list
        return proportions_df, high_impact_list
    
    def analyze_source_impact(self, impact_data):
        """
        Analyze the impact of sources using external H-index or Impact Factor data.

        Parameters:
            impact_data (pd.DataFrame): External dataset with columns ["Source", "H-Index", "Impact Factor"].

        Returns:
            pd.DataFrame: Sources ranked by H-index and Impact Factor with article counts.
        """
        source_counts = []

        for entry in self.processed_data:
            source = entry['bibliographic_metadata'].get('publisher', '').strip().lower()
            if source:
                source_counts.append(source)

        source_frequency = Counter(source_counts)
        source_df = pd.DataFrame(source_frequency.items(), columns=["Source", "Article Count"])

        # Merge with external impact data
        impact_data["Source"] = impact_data["Source"].str.lower()
        merged_df = source_df.merge(impact_data, on="Source", how="left")

        # Sort by H-index or Impact Factor
        ranked_sources = merged_df.sort_values(by=["H-Index", "Impact Factor"], ascending=False)
        return ranked_sources   
# --------------------------------------------------------------
# -- Visualization --------------------------------------------
# --------------------------------------------------------------      
class Visualizer:
    """Handles visualization of author analysis results."""

    @staticmethod
    def plot_bar_chart(df, title, filename, x_col, y_col):
        """Generic method to plot bar charts."""

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[x_col], df[y_col], color="#87CEEB", edgecolor="black")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"../output/m3_sources/{filename}.svg", format="svg")
        fig.savefig(f"../output/m3_sources/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_pie_chart(df, title, filename, labels_col, values_col):
        """Plot a chart for proportions."""

        # Validate column names
        if labels_col not in df.columns or values_col not in df.columns:
            raise ValueError(f"Columns '{labels_col}' and '{values_col}' must exist in the DataFrame")

        # Create the pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(
            df[values_col].values,  # Use .values to avoid index confusion
            labels=df[labels_col].values,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#87CEEB", "#FFCCCB"]
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"../output/m3_sources/{filename}.svg", format="svg")
        fig.savefig(f"../output/m3_sources/{filename}.png", format="png")
        plt.show()

# --------------------------------------------------------------
# -- Reporting -----------------------------------------------
# --------------------------------------------------------------
class Reporter:
    """Handles reporting of author analysis results."""

    @staticmethod
    def save_to_csv(df, filename):
        """Save a DataFrame to a CSV file."""
        output_path = f"../output/m2_authors/{filename}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
