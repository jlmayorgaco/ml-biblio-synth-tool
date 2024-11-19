# src/eda.py

from collections import Counter
import pandas as pd

class EDA:
    """Class for performing exploratory data analysis on keywords and text data."""

    def __init__(self, processed_data):
        self.processed_data = processed_data

    def extract_keywords(self):
        """Extracts keywords from the references and returns a flat list."""
        keywords_list = []
        for ref, _ in self.processed_data:
            keywords = ref.get("keywords", "").split(", ")
            keywords_list.extend([kw.strip().lower() for kw in keywords])
        return keywords_list

    def get_keyword_frequencies(self, keywords_list):
        """Returns keyword frequencies in descending order."""
        return Counter(keywords_list)

    def keyword_frequency_df(self, top_n=20):
        """Returns a DataFrame of top N keywords for analysis."""
        keywords = self.extract_keywords()
        freq = self.get_keyword_frequencies(keywords)
        df = pd.DataFrame(freq.most_common(top_n), columns=['Keyword', 'Frequency'])
        return df
