import re
import string
from nltk.corpus import stopwords
from collections import Counter

class WordProcessor:
    """Handles text cleaning and filtering operations."""

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def clean_and_filter_words(self, text):
        """
        Clean and filter words by removing stop words, punctuation, single characters, and numbers.
        Args:
            text (str): Input text to clean.
        Returns:
            list: List of cleaned and filtered words.
        """
        words = text.split()
        return [
            word.lower() for word in words
            if word.lower() not in self.stop_words
            and word.lower() != "cid"  # Exclude unwanted terms like "cid"
            and not word.isdigit()
            and len(word) > 1
            and not all(char in string.punctuation for char in word)
        ]

