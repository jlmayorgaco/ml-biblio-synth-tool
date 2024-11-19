# src/visualization.py

import matplotlib.pyplot as plt
from wordcloud import WordCloud

class Visualizer:
    """Class for creating and saving visualizations of keywords and text data."""

    @staticmethod
    def plot_keyword_bar(df, save_path=None):
        """Plots a bar chart of keyword frequencies."""
        plt.figure(figsize=(10, 6))
        plt.barh(df['Keyword'], df['Frequency'], color='skyblue')
        plt.xlabel('Frequency')
        plt.ylabel('Keyword')
        plt.title('Top Keywords by Frequency')
        plt.gca().invert_yaxis()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    @staticmethod
    def generate_word_cloud(text, save_path=None):
        """Generates and displays a word cloud from the input text."""
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
        plt.show()
