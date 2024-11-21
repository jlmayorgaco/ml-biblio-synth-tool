from collections import Counter
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('stopwords')

# Get the list of English stop words
stop_words = set(stopwords.words('english'))

# Update style for a professional look
plt.style.use('default')
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.figsize": (14, 6),
    "axes.edgecolor": "black",
    "grid.color": "gray",
    "grid.linestyle": "--",
    "grid.linewidth": 0.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white"
})


class EDA:
    """Class for performing exploratory data analysis on bibliographic metadata and plain text."""

    def __init__(self, processed_data):
        self.processed_data = processed_data

    def clean_and_filter_words(self, text):
        """Clean and filter words by removing stop words, punctuation, single characters, and specific terms."""
        words = text.split()
        return [
            word.lower() for word in words
            if word.lower() not in stop_words  # Exclude stop words
            and word.lower() != "cid"  # Exclude the 'cid' term explicitly
            and not word.isdigit()  # Exclude numbers
            and len(word) > 1  # Exclude single characters
            and not all(char in string.punctuation for char in word)  # Exclude punctuation-only strings
        ]

    def word_frequency_df(self):
        """Generates a DataFrame with word frequencies."""
        word_counter = Counter()
        for entry in self.processed_data:
            filtered_words = self.clean_and_filter_words(entry["plain_text"])
            word_counter.update(filtered_words)
        word_freq_df = pd.DataFrame(word_counter.items(), columns=['Word', 'Frequency'])
        return word_freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)

    def completeness_metrics(self):
        """Calculates completeness metrics for each field."""
        essential_fields = ['title', 'author', 'year', 'doi']
        completeness_data = []
        for field in essential_fields:
            filled_count = sum(1 for entry in self.processed_data if entry['bibliographic_metadata'].get(field))
            completeness_percentage = (filled_count / len(self.processed_data)) * 100
            completeness_data.append({'Field': field, 'Filled': filled_count, 'Completeness (%)': completeness_percentage})
        return pd.DataFrame(completeness_data)

    def plot_completeness(self, completeness_df):
        """Plots completeness of fields as a bar and pie chart side by side."""
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))
        ax1.barh(completeness_df['Field'], completeness_df['Completeness (%)'], color='#a6cee3')
        ax1.set_xlabel('Completeness (%)', labelpad=10)
        ax1.set_title('Completeness of Fields', pad=12, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
        ax1.axvline(x=80, color='red', linestyle=':', linewidth=1.5, label='80% Threshold')
        ax1.legend(loc='lower right')
        ax2.pie(completeness_df['Filled'], labels=completeness_df['Field'], autopct='%1.1f%%',
                startangle=140, colors=['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'])
        ax2.set_title('Proportion of Non-Missing Fields', pad=15, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def identify_missing_fields(self):
        """Identifies which records have missing fields and returns a summary."""
        essential_fields = ["author", "year", "title", "doi"]
        missing_fields_summary = []
        for i, entry in enumerate(self.processed_data):
            missing_fields = [field for field in essential_fields if not entry["bibliographic_metadata"].get(field)]
            if missing_fields:
                missing_fields_summary.append({"Record": i + 1, "Missing Fields": missing_fields})
        return pd.DataFrame(missing_fields_summary)

    def plot_missing_fields(self, missing_fields_df):
        """Plots missing fields as a bar and pie chart side by side."""
        all_missing_fields = [field for fields in missing_fields_df['Missing Fields'] for field in fields]
        missing_field_counts = Counter(all_missing_fields)
        fields, counts = zip(*missing_field_counts.most_common())
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.bar(fields, counts, color='#87CEEB')
        ax1.set_title('Frequency of Missing Fields', pad=12, fontweight='bold')
        ax1.set_xlabel('Missing Field')
        ax1.set_ylabel('Count')
        ax2.pie(counts, labels=fields, autopct='%1.1f%%', startangle=140,
                colors=['#87CEEB', '#FFA07A', '#FFD700'])
        ax2.set_title('Proportion of Missing Fields', pad=12, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def generate_metadata_summary_df(self):
        """Creates a DataFrame with item ID, metadata, and word count of the plain text."""
        data_summary = []
        for entry in self.processed_data:
            metadata = entry['bibliographic_metadata']
            plain_text = entry['plain_text']
            word_count = len(plain_text.split())
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

    def analyze_metadata_summary(self, metadata_summary_df):
        """Analyzes metadata summary DataFrame."""
        top_5_longest = metadata_summary_df.nlargest(5, 'Word Count')
        plt.figure(figsize=(10, 6))
        plt.bar(top_5_longest['ID'], top_5_longest['Word Count'], color='skyblue', edgecolor='black')
        plt.title('Top 5 Records by Word Count in Plain Text')
        plt.xlabel('Record ID')
        plt.ylabel('Word Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        print("\nTop 5 Records with Highest Word Count:")
        print(top_5_longest[['ID', 'Title', 'Word Count']])

    def analyze_publication_trends(self, metadata_summary_df):
        """Plots temporal distribution of publications."""
        metadata_summary_df['Year'] = pd.to_numeric(metadata_summary_df['Year'], errors='coerce')
        publication_trends = metadata_summary_df.groupby('Year').size().reset_index(name='Publication Count')
        publication_trends = publication_trends.dropna()
        plt.figure(figsize=(12, 6))
        plt.plot(publication_trends['Year'], publication_trends['Publication Count'], marker='o', color='blue', label='Publications')
        plt.title('Temporal Distribution of Publications', pad=12, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Number of Publications')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        significant_events = {2020: 'COVID-19 Pandemic', 2008: 'Global Financial Crisis', 1990: 'Start of Digital Revolution'}
        for year, event in significant_events.items():
            if year in publication_trends['Year'].values:
                plt.axvline(x=year, color='red', linestyle='--', linewidth=1, alpha=0.7)
                plt.text(year, max(publication_trends['Publication Count']) * 0.8, event, rotation=90,
                         verticalalignment='center', color='red', fontsize=9)
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
        print("\nPublication Trends Summary:")
        print(publication_trends)

    def term_frequency_by_year(self, n=5):
        """Calculates term frequency by year."""
        all_terms = [
            word for entry in self.processed_data
            for word in self.clean_and_filter_words(entry['plain_text'])
        ]
        term_counts = Counter(all_terms).most_common(n)
        top_terms = [term for term, _ in term_counts]
        data = []
        for entry in self.processed_data:
            year = entry["bibliographic_metadata"].get("year", "Unknown")
            terms_in_text = Counter(self.clean_and_filter_words(entry['plain_text']))
            term_frequencies = {term: terms_in_text.get(term, 0) for term in top_terms}
            term_frequencies["year"] = year
            data.append(term_frequencies)
        df = pd.DataFrame(data)
        term_columns = [col for col in df.columns if col != "year"]
        df = df.groupby("year")[term_columns].sum().sort_index()
        return df

    def plot_term_trends(self, df):
        """Plots term trends over time."""
        plt.figure(figsize=(10, 6))
        for term in df.columns:
            plt.plot(df.index, df[term], marker='o', label=term)
        plt.title("Trend of Top Terms Over Time (Filtered)", fontsize=14, fontweight="bold")
        plt.xlabel("Year", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(title="Terms", fontsize=10)
        plt.grid(axis="y", linestyle="--", color="gray", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def word_frequency_df_with_cloud(self, top_n=20):
        """
        Generates a DataFrame with word frequencies and plots a word cloud.
        
        Args:
            top_n (int): Number of top words to include in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame of word frequencies sorted in descending order.
        """
        # Compute word frequencies
        word_counter = Counter()
        for entry in self.processed_data:
            filtered_words = self.clean_and_filter_words(entry["plain_text"])
            word_counter.update(filtered_words)
        
        # Create the DataFrame
        word_freq_df = pd.DataFrame(word_counter.items(), columns=['Word', 'Frequency'])
        word_freq_df = word_freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
        
        # Plot word cloud
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color="white"
        ).generate_from_frequencies(dict(word_counter))
        
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Word Cloud of Most Frequent Terms", fontsize=16, fontweight="bold")
        plt.show()
        
        # Return the top_n words
        return word_freq_df.head(top_n)
    

    def extract_keywords(self):
        """Extract keywords from metadata or fallback to title/abstract."""
        all_keywords = []
        for entry in self.processed_data:
            metadata = entry["bibliographic_metadata"]
            keywords = metadata.get("keywords", "")

            if not keywords:  # If no keywords are available, fallback to title or abstract
                keywords = metadata.get("title", "") + " " + metadata.get("abstract", "")

            if keywords:
                # Split keywords by common delimiters and clean
                keyword_list = re.split(r'[;,.\s]', keywords)
                all_keywords.extend(self.clean_and_filter_words(" ".join(keyword_list)))
            else:
                print(f"No keywords, title, or abstract found for record: {metadata.get('title', 'Unknown')}")
        
        print(f"Extracted {len(all_keywords)} keywords from the dataset.")
        return Counter(all_keywords)

    def plot_keyword_frequency(self, top_n=10):
        """Plot the top N keywords by frequency."""
        keyword_counts = self.extract_keywords()
        top_keywords = keyword_counts.most_common(top_n)

        # Prepare data for plotting
        keywords, counts = zip(*top_keywords)

        # Plot bar chart
        plt.figure(figsize=(10, 6))
        plt.barh(keywords, counts, color='skyblue', edgecolor='black')
        plt.title(f'Top {top_n} Keywords by Frequency', fontweight='bold')
        plt.xlabel('Frequency')
        plt.ylabel('Keywords')
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def keyword_trends_over_time(self, top_n=5):
        """Analyze trends of top N keywords over time."""
        keyword_counts = self.extract_keywords()
        top_keywords = [keyword for keyword, _ in keyword_counts.most_common(top_n)]

        # Initialize trends dictionary
        trends = {keyword: [] for keyword in top_keywords}
        trends["year"] = []

        # Group by year
        for entry in self.processed_data:
            year = entry["bibliographic_metadata"].get("year", "Unknown")
            if year == "Unknown":
                continue
            year = int(year)  # Ensure year is numeric
            text = entry["plain_text"]
            words = self.clean_and_filter_words(text)
            word_counts = Counter(words)
            for keyword in top_keywords:
                trends[keyword].append(word_counts.get(keyword, 0))
            trends["year"].append(year)

        # Convert to DataFrame for plotting
        _trends_df = pd.DataFrame(trends)
        trends_df = _trends_df.groupby("year").sum().sort_index().head(10)

        # Plot trends
        plt.figure(figsize=(10, 6))
        for keyword in top_keywords:
            plt.plot(trends_df.index, trends_df[keyword], marker='o', label=keyword)
        plt.title(f'Trends of Top {top_n} Keywords Over Time', fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Frequency')
        plt.legend(title="Keywords", fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        return trends_df

    def identify_emerging_and_declining_keywords(self, period=5):
        """
        Identify emerging and declining keywords based on growth rate over the specified period.
        Emerging keywords: Positive growth rate in the last n years.
        Declining keywords: Negative growth rate in the last n years.
        """
        # Extract keywords and their frequencies by year
        trends_df = self.keyword_trends_over_time(top_n = 10000)

        print(' trends_df ')
        print(trends_df)

        if trends_df.empty:
            print("No keyword trends data available.")
            return None, None

        # Focus only on the last `period` years
        recent_years = trends_df.index.max() - period
        recent_data = trends_df[trends_df.index > recent_years]
        previous_data = trends_df[trends_df.index <= recent_years]

        # Calculate growth rate for each term
        recent_sums = recent_data.sum()
        previous_sums = previous_data.sum()
        growth_rate = ((recent_sums - previous_sums) / (previous_sums + 1e-6)) * 100  # Avoid division by zero

        # Debug print statements
        print("\n--- Debugging Growth Rate Calculation ---")
        print("Recent Sums (last n years):")
        print(recent_sums)
        print("\nPrevious Sums (before last n years):")
        print(previous_sums)
        print("\nGrowth Rate (%):")
        print(growth_rate)

        # Separate emerging and declining keywords
        emerging = growth_rate[growth_rate > 0].sort_values(ascending=False)
        declining = growth_rate[growth_rate < 0].sort_values(ascending=True)

        # Debug print statements for emerging and declining keywords
        print("\nEmerging Keywords:")
        print(emerging)
        print("\nDeclining Keywords:")
        print(declining)

        return emerging, declining


    def plot_emerging_declining_keywords(self, period=5):
        """
        Plot emerging and declining keywords based on growth rate.
        """
        _emerging, _declining = self.identify_emerging_and_declining_keywords(period)

        emerging = _emerging.head(10)
        declining = _declining.head(10)

        if emerging.empty and declining.empty:
            print("Not enough data to identify emerging and declining keywords.")
            return

        # Bar chart for emerging and declining keywords
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Emerging keywords
        if not emerging.empty:
            ax1.barh(emerging.index, emerging.values, color='green', edgecolor='black')
            ax1.set_title('Emerging Keywords', fontweight='bold')
            ax1.set_xlabel('Growth Rate (%)')
            ax1.grid(axis='x', linestyle='--', alpha=0.7)

        # Declining keywords
        if not declining.empty:
            ax2.barh(declining.index, declining.values, color='red', edgecolor='black')
            ax2.set_title('Declining Keywords', fontweight='bold')
            ax2.set_xlabel('Growth Rate (%)')
            ax2.grid(axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()