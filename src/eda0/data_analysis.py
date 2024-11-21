import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from .completeness_analysis import CompletenessAnalysis
from .word_processing import WordProcessor


class DataAnalysis:
    """Performs high-level data analysis."""

    def __init__(self, processed_data, word_processor):
        self.processed_data = processed_data
        self.word_processor = word_processor

    def word_frequency(self):
        """
        Calculate word frequencies from plain text.
        Returns:
            pd.DataFrame: DataFrame of word frequencies.
        """
        word_counter = Counter()
        for entry in self.processed_data:
            filtered_words = self.word_processor.clean_and_filter_words(entry["plain_text"])
            word_counter.update(filtered_words)
        return pd.DataFrame(word_counter.items(), columns=['Word', 'Frequency']).sort_values(by="Frequency", ascending=False).reset_index(drop=True)

    def plot_word_cloud(self, word_freq_df):
        """
        Plot a word cloud of the most frequent words.
        Args:
            word_freq_df (pd.DataFrame): DataFrame containing word frequencies.
        """
        word_dict = dict(zip(word_freq_df['Word'], word_freq_df['Frequency']))
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_dict)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Most Frequent Terms", fontsize=16, fontweight="bold")
        plt.show()

    def analyze_keyword_trends(self, top_n=10):
        """
        Analyze and plot keyword trends over time.
        Args:
            top_n (int): Number of top keywords to analyze.
        Returns:
            pd.DataFrame: DataFrame of keyword trends over years.
        """
        all_terms = [
            word for entry in self.processed_data
            for word in self.word_processor.clean_and_filter_words(entry["plain_text"])
        ]
        term_counts = Counter(all_terms).most_common(top_n)
        top_terms = [term for term, _ in term_counts]

        trends = {term: [] for term in top_terms}
        trends["year"] = []
        for entry in self.processed_data:
            year = entry["bibliographic_metadata"].get("year", "Unknown")
            if year == "Unknown":
                continue
            year = int(year)
            text = entry["plain_text"]
            words = self.word_processor.clean_and_filter_words(text)
            word_counts = Counter(words)
            for term in top_terms:
                trends[term].append(word_counts.get(term, 0))
            trends["year"].append(year)

        trends_df = pd.DataFrame(trends).groupby("year").sum().sort_index()

        # Plot trends
        plt.figure(figsize=(10, 6))
        for term in top_terms:
            plt.plot(trends_df.index, trends_df[term], marker='o', label=term)
        plt.title(f"Trends of Top {top_n} Keywords Over Time", fontweight="bold")
        plt.xlabel("Year")
        plt.ylabel("Frequency")
        plt.legend(title="Keywords", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        return trends_df

    def identify_emerging_and_declining_keywords(self, period=5):
        """
        Identify emerging and declining keywords based on growth rates.
        Args:
            period (int): Number of recent years to consider.
        Returns:
            tuple: (emerging_keywords, declining_keywords)
        """
        trends_df = self.analyze_keyword_trends(top_n=10000)
        if trends_df.empty:
            return None, None

        recent_years = trends_df.index.max() - period
        recent_data = trends_df[trends_df.index > recent_years]
        previous_data = trends_df[trends_df.index <= recent_years]

        recent_sums = recent_data.sum()
        previous_sums = previous_data.sum()
        growth_rate = ((recent_sums - previous_sums) / (previous_sums + 1e-6)) * 100

        emerging = growth_rate[growth_rate > 0].sort_values(ascending=False)
        declining = growth_rate[growth_rate < 0].sort_values(ascending=True)

        return emerging, declining

    def plot_emerging_and_declining_keywords(self, period=5):
        """
        Plot emerging and declining keywords.
        Args:
            period (int): Number of recent years to consider.
        """
        emerging, declining = self.identify_emerging_and_declining_keywords(period)

        if emerging.empty and declining.empty:
            print("Not enough data to identify emerging and declining keywords.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Emerging keywords
        if not emerging.empty:
            ax1.barh(emerging.index, emerging.values, color='green', edgecolor='black')
            ax1.set_title("Emerging Keywords", fontweight="bold")
            ax1.set_xlabel("Growth Rate (%)")
            ax1.grid(axis="x", linestyle="--", alpha=0.7)

        # Declining keywords
        if not declining.empty:
            ax2.barh(declining.index, declining.values, color='red', edgecolor='black')
            ax2.set_title("Declining Keywords", fontweight="bold")
            ax2.set_xlabel("Growth Rate (%)")
            ax2.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()

    def generate_metadata_summary(self):
        """Creates a DataFrame with item ID, metadata, and word count of the plain text."""
        data_summary = []
        for entry in self.processed_data:
            metadata = entry['bibliographic_metadata']
            plain_text = entry['plain_text']
            word_count = len(plain_text.split())  # Count words in plain text

            summary_row = {
                "ID": metadata.get('ID', 'N/A'),
                "DOI": metadata.get('doi', 'N/A'),
                "Author": metadata.get('author', 'N/A'),
                "Year": metadata.get('year', 'N/A'),
                "Title": metadata.get('title', 'N/A'),
                "Publisher": metadata.get('publisher', 'N/A'),
                "Word Count": word_count
            }
            data_summary.append(summary_row)

        return pd.DataFrame(data_summary)
    
    def plot_top_5_records_by_word_count(self, metadata_summary_df):
        """Plots the top 5 records with the highest word count in plain text."""
        # Identify the top 5 records by word count
        top_5_longest = metadata_summary_df.nlargest(5, 'Word Count')

        # Plot the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(top_5_longest['ID'], top_5_longest['Word Count'], color='skyblue', edgecolor='black')
        plt.title('Top 5 Records by Word Count in Plain Text', fontweight='bold')
        plt.xlabel('Record ID')
        plt.ylabel('Word Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Print the top 5 records for reference
        print("\nTop 5 Records with Highest Word Count:")
        print(top_5_longest[['ID', 'Title', 'Word Count']])


    def analyze_publication_trends(self):
        """Plots the temporal distribution of publications."""
        # Create a DataFrame for metadata summary
        metadata_summary_df = CompletenessAnalysis(self.processed_data).generate_metadata_summary()

        # Convert Year column to numeric and drop invalid values
        metadata_summary_df['Year'] = pd.to_numeric(metadata_summary_df['Year'], errors='coerce')
        publication_trends = metadata_summary_df.groupby('Year').size().reset_index(name='Publication Count')
        publication_trends = publication_trends.dropna()  # Drop rows with NaN year

        # Plot the time series
        plt.figure(figsize=(12, 6))
        plt.plot(publication_trends['Year'], publication_trends['Publication Count'], marker='o', color='blue', label='Publications')
        plt.title('Temporal Distribution of Publications', pad=12, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate significant events
        significant_events = {
            2020: 'COVID-19 Pandemic',
            2008: 'Global Financial Crisis',
            1990: 'Start of Digital Revolution'
        }
        for year, event in significant_events.items():
            if year in publication_trends['Year'].values:
                plt.axvline(x=year, color='red', linestyle='--', linewidth=1, alpha=0.7)
                plt.text(year, max(publication_trends['Publication Count']) * 0.8, event, rotation=90,
                         verticalalignment='center', color='red', fontsize=9)

        # Show legend
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()

        # Print summary
        print("\nPublication Trends Summary:")
        print(publication_trends)
