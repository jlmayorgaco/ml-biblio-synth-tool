import re
import itertools

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from textblob import TextBlob
from collections import Counter, defaultdict
from networkx.algorithms.community import greedy_modularity_communities

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

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

    def get_most_frequent_quotes(self, n=10):
        """
        Extract and count the most frequently used quotes in the dataset.

        Parameters:
            n (int): Number of top quotes to return.

        Returns:
            pd.DataFrame: A DataFrame with the top quotes and their frequencies.
        """
        quote_counts = Counter()

        for entry in self.processed_data:
            text = entry.get("plain_text", "")
            # Use regex to extract text within quotes
            quotes = re.findall(r'"(.*?)"', text)
            quote_counts.update(quotes)

        # Convert to DataFrame and sort
        quotes_df = pd.DataFrame(quote_counts.items(), columns=["Quote", "Frequency"])
        return quotes_df.sort_values(by="Frequency", ascending=False).head(n)

    def get_quotes_by_year(self):
        """
        Analyze how quotes are distributed over time.

        Returns:
            pd.DataFrame: A DataFrame with quotes, years, and their frequencies.
        """
        quote_counts = []

        for entry in self.processed_data:
            text = entry.get("plain_text", "")
            year = entry["bibliographic_metadata"].get("year", None)

            # Validate year
            try:
                year = int(year)
            except (ValueError, TypeError):
                continue

            # Extract quotes using regex
            quotes = re.findall(r'"(.*?)"', text)
            for quote in quotes:
                quote_counts.append({"Year": year, "Quote": quote})

        # Convert to DataFrame
        quotes_df = pd.DataFrame(quote_counts)

        # Group by quote and year to count occurrences
        grouped = quotes_df.groupby(["Quote", "Year"]).size().reset_index(name="Frequency")

        return grouped

    def get_quote_contexts(self, quotes_to_analyze, context_window=2):
        """
        Extract the context of specified quotes from the dataset.

        Parameters:
            quotes_to_analyze (list): List of quotes to analyze.
            context_window (int): Number of sentences before and after the quote to include as context.

        Returns:
            pd.DataFrame: A DataFrame with the quotes, their contexts, and associated metadata.
        """
        contexts = []

        for entry in self.processed_data:
            text = entry.get("plain_text", "")
            title = entry["bibliographic_metadata"].get("title", "")
            authors = entry["bibliographic_metadata"].get("author", "")
            year = entry["bibliographic_metadata"].get("year", None)

            # Split the text into sentences
            sentences = re.split(r"(?<=[.!?]) +", text)

            for quote in quotes_to_analyze:
                for i, sentence in enumerate(sentences):
                    if quote in sentence:
                        # Extract context window
                        start = max(0, i - context_window)
                        end = min(len(sentences), i + context_window + 1)
                        context = " ".join(sentences[start:end])

                        # Append context information
                        contexts.append({
                            "Quote": quote,
                            "Context": context,
                            "Title": title,
                            "Authors": authors,
                            "Year": year
                        })

        return pd.DataFrame(contexts)

    def analyze_sentiment_of_contexts(self, contexts_df):
        """
        Perform sentiment analysis on the context of quotes.

        Parameters:
            contexts_df (pd.DataFrame): DataFrame with columns ["Quote", "Context", ...].

        Returns:
            pd.DataFrame: A DataFrame with sentiment scores and classifications for each quote context.
        """
        contexts_df = contexts_df.copy()
        
        def get_sentiment(context):
            sentiment = TextBlob(context).sentiment
            return sentiment.polarity, sentiment.subjectivity

        # Analyze sentiment
        contexts_df["Sentiment Polarity"], contexts_df["Sentiment Subjectivity"] = zip(
            *contexts_df["Context"].apply(get_sentiment)
        )
        contexts_df["Sentiment Classification"] = contexts_df["Sentiment Polarity"].apply(
            lambda x: "Positive" if x > 0.1 else "Negative" if x < -0.1 else "Neutral"
        )
        
        return contexts_df

    def extract_themes_from_contexts(self, contexts_df, top_n=10):
        """
        Extract themes (keywords) from the contexts of quotes.

        Parameters:
            contexts_df (pd.DataFrame): DataFrame with a "Context" column.
            top_n (int): Number of top keywords to extract.

        Returns:
            pd.DataFrame: A DataFrame with keywords and their frequencies.
        """
        vectorizer = CountVectorizer(stop_words="english", max_features=top_n)
        context_texts = contexts_df["Context"].tolist()
        
        # Fit and transform contexts to extract keywords
        X = vectorizer.fit_transform(context_texts)
        keywords = vectorizer.get_feature_names_out()
        frequencies = X.sum(axis=0).A1

        # Create a DataFrame with keywords and their frequencies
        themes_df = pd.DataFrame({"Keyword": keywords, "Frequency": frequencies}).sort_values(by="Frequency", ascending=False)
        return themes_df

    @staticmethod
    def plot_word_cloud_from_themes(df, title, filename):
        """
        Plot a word cloud from extracted themes.

        Parameters:
            df (pd.DataFrame): DataFrame with columns ["Keyword", "Frequency"].
            title (str): Title of the plot.
            filename (str): Name of the file to save the plot.
        """
        word_freq = dict(zip(df["Keyword"], df["Frequency"]))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"../output/m4_quotes/{filename}.svg", format="svg")
        fig.savefig(f"../output/m4_quotes/{filename}.png", format="png")
        plt.show()
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
    def plot_most_frequent_quotes(df, title, filename):
        """Plot a bar chart of the most frequent quotes."""
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.barh(df["Quote"], df["Frequency"], color="#87CEEB")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Frequency", fontsize=12)
        ax.set_ylabel("Quotes", fontsize=12)
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"../output/m4_quotes/{filename}.svg", format="svg")
        fig.savefig(f"../output/m4_quotes/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_quote_trends(df, title, filename):
        """
        Plot trends of quotes over time.

        Parameters:
            df (pd.DataFrame): DataFrame with columns ["Quote", "Year", "Frequency"].
            title (str): Title of the plot.
            filename (str): Name of the file to save the plot.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot each quote's trend over time
        for quote in df["Quote"].unique():
            subset = df[df["Quote"] == quote]
            ax.plot(subset["Year"], subset["Frequency"], label=quote, marker="o")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Year", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(title="Quotes", fontsize=10)
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"../output/m4_quotes/{filename}.svg", format="svg")
        fig.savefig(f"../output/m4_quotes/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_sentiment_distribution(df, title, filename):
        """
        Plot the sentiment distribution of quote contexts.

        Parameters:
            df (pd.DataFrame): DataFrame with a "Sentiment Classification" column.
            title (str): Title of the plot.
            filename (str): Name of the file to save the plot.
        """
        sentiment_counts = df["Sentiment Classification"].value_counts()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(
            sentiment_counts,
            labels=sentiment_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#87CEEB", "#FFCCCB", "#98FB98"]
        )
        ax.set_title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        # Save the plot
        fig.savefig(f"../output/m4_quotes/{filename}.svg", format="svg")
        fig.savefig(f"../output/m4_quotes/{filename}.png", format="png")
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
