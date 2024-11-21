import re
import itertools

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

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
    def get_top_authors(self, n=10):
        """Get the top N authors based on the number of documents."""
        all_authors = []
        for entry in self.processed_data:
            # Extract the 'author' field from bibliographic metadata
            authors = entry['bibliographic_metadata'].get('author', '')
            if authors:
                # Split on commas and 'and' to separate individual authors
                authors_list = [author.strip() for part in authors.split(",") for author in part.split(" and ")]
                all_authors.extend(authors_list)

        # Count occurrences of each author
        author_counts = Counter(all_authors)

        # Return the top N authors as a DataFrame
        return pd.DataFrame(author_counts.most_common(n), columns=["Author", "Count"])

    def get_author_trends(self, authors_to_track):
        """Get trends of specific authors over time."""
        trends = []
        for entry in self.processed_data:
            year = self.parse_year(entry['bibliographic_metadata'].get('year'))
            authors = entry['bibliographic_metadata'].get('author', '')
            if not year or not authors:
                continue

            for author in authors.split(", "):
                if author in authors_to_track:
                    trends.append({"Year": year, "Author": author, "Count": 1})

        trends_df = pd.DataFrame(trends)
        return trends_df.groupby(['Year', 'Author'], as_index=False).sum()

    def get_most_collaborative_authors(self, n=10):
        """Get the top N most collaborative authors by unique co-authors."""
        collaboration_counts = Counter()

        for entry in self.processed_data:
            authors = entry['bibliographic_metadata'].get('author', '')
            if authors:
                # Split authors and calculate pairwise collaborations
                author_list = [author.strip() for part in authors.split(",") for author in part.split(" and ")]
                for author in author_list:
                    collaboration_counts[author] += len(author_list) - 1  # Exclude self from count

        return pd.DataFrame(collaboration_counts.most_common(n), columns=["Author", "Unique Co-Authors"])
    
    def create_collaboration_network(self):
        """Create a co-authorship network."""
        collaboration_counts = Counter()

        for entry in self.processed_data:
            authors = entry['bibliographic_metadata'].get('author', '')
            if authors:
                # Split authors and calculate pairwise collaborations
                author_list = [author.strip() for part in authors.split(",") for author in part.split(" and ")]
                for pair in itertools.combinations(set(author_list), 2):  # Unique pairs
                    collaboration_counts[pair] += 1

        # Convert to DataFrame
        collaboration_data = pd.DataFrame([
            {"Author1": pair[0], "Author2": pair[1], "Weight": count}
            for pair, count in collaboration_counts.items()
        ])
        return collaboration_data
    
    def extract_clusters(self, collaboration_data, filename="author_clusters"):
        """Extract clusters (communities) from the co-authorship network and save to a table."""

        # Create the graph
        G = nx.Graph()

        # Add edges with weights
        for _, row in collaboration_data.iterrows():
            G.add_edge(row["Author1"], row["Author2"], weight=row["Weight"])

        # Detect communities
        communities = list(greedy_modularity_communities(G))

        # Prepare the clusters table
        clusters_table = pd.DataFrame({
            "Cluster ID": [f"Cluster {i+1}" for i in range(len(communities))],
            "Authors": [", ".join(sorted(community)) for community in communities]
        })

        # Save the table
        Reporter.save_to_csv(clusters_table, filename)
        
        print(f"Clusters saved to ../output/m2_authors/{filename}.csv")
        return clusters_table
    
    def get_productivity_distribution(self):
        """Get the distribution of document counts per author."""
        all_authors = []
        for entry in self.processed_data:
            authors = entry['bibliographic_metadata'].get('author', '')
            if authors:
                # Split authors into individual names
                authors_list = [author.strip() for part in authors.split(",") for author in part.split(" and ")]
                all_authors.extend(authors_list)

        # Count occurrences of each author
        author_counts = Counter(all_authors)

        # Create a DataFrame
        productivity_df = pd.DataFrame(author_counts.items(), columns=["Author", "Document Count"])
        return productivity_df

    def get_longest_standing_authors(self, n=10):
        """Get the top N longest-standing authors based on their active publication spans."""
        author_years = defaultdict(list)

        # Collect publication years for each author
        for entry in self.processed_data:
            authors = entry['bibliographic_metadata'].get('author', '')
            year = entry['bibliographic_metadata'].get('year', None)

            if authors and year:
                try:
                    year = int(year)  # Convert year to integer
                except ValueError:
                    continue  # Skip invalid years

                # Split authors into individual names
                authors_list = [author.strip() for part in authors.split(",") for author in part.split(" and ")]
                for author in authors_list:
                    author_years[author].append(year)

        # Calculate active spans for each author
        longest_standing_authors = [
            {
                "Author": author,
                "First Publication": min(years),
                "Last Publication": max(years),
                "Active Span (Years)": max(years) - min(years) + 1
            }
            for author, years in author_years.items()
        ]

        # Sort by active span and select the top N authors
        longest_standing_authors = sorted(longest_standing_authors, key=lambda x: x["Active Span (Years)"], reverse=True)[:n]
        return pd.DataFrame(longest_standing_authors)

    def get_author_term_relationships(self):
        """Analyze how each author relates to specific terms based on co-occurrences."""
        import itertools
        from collections import Counter

        # List to store all relationships
        author_term_data = []

        for entry in self.processed_data:
            authors = entry['bibliographic_metadata'].get('author', '')
            plain_text = entry.get('plain_text', '')

            if authors and plain_text:
                # Split authors into individual names
                authors_list = [author.strip() for part in authors.split(",") for author in part.split(" and ")]

                # Clean and tokenize the text
                terms = self._clean_text(plain_text)

                # Calculate co-occurrences for each author
                cooccurrences = Counter(itertools.combinations(terms, 2))
                for author in authors_list:
                    for (term1, term2), count in cooccurrences.items():
                        author_term_data.append({
                            "Author": author,
                            "Term1": term1,
                            "Term2": term2,
                            "Co-occurrence Count": count
                        })

        # Return the collected data as a DataFrame
        return pd.DataFrame(author_term_data)
    @staticmethod
    def parse_year(year):
        """Convert year to integer if valid, otherwise return None."""
        try:
            return int(year) if int(year) > 0 else None
        except (ValueError, TypeError):
            return None

# --------------------------------------------------------------
# -- Visualization --------------------------------------------
# --------------------------------------------------------------      
class Visualizer:
    """Handles visualization of author analysis results."""

    @staticmethod
    def plot_bar_chart(df, title, filename, x_col="Author", y_col="Count"):
        """Generic method to plot bar charts."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(df[x_col], df[y_col], color="#87CEEB")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        fig.savefig(f"../output/m2_authors/{filename}.svg", format="svg")
        fig.savefig(f"../output/m2_authors/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_line_chart(df, title, filename, x_col="Year", y_col="Count", hue_col="Author"):
        """Plot a line chart for trends over time, with each line representing a different author."""
        fig, ax = plt.subplots(figsize=(12, 6))
        for author in df[hue_col].unique():
            author_data = df[df[hue_col] == author]
            ax.plot(author_data[x_col], author_data[y_col], marker='o', label=author)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.legend(title=hue_col, fontsize=10, loc="upper left")
        ax.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        fig.savefig(f"../output/m2_authors/{filename}.svg", format="svg")
        fig.savefig(f"../output/m2_authors/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_author_collaboration_network(collaboration_data, filename="author_collaboration_network"):
        """Plot a co-authorship network with improved visual settings."""

        # Create the graph
        G = nx.Graph()

        # Add edges with weights
        for _, row in collaboration_data.iterrows():
            G.add_edge(row["Author1"], row["Author2"], weight=row["Weight"])

        # Calculate node size (degree centrality)
        node_sizes = [1000 * nx.degree_centrality(G)[node] for node in G.nodes()]

        # Detect communities and assign colors
        communities = list(greedy_modularity_communities(G))
        community_colors = {node: i for i, community in enumerate(communities) for node in community}
        node_colors = [community_colors[node] for node in G.nodes()]

        # Generate layout
        pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(15, 15))

        # Draw nodes with size and color
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.tab10,  # Use a colormap for clusters
            alpha=0.9
        )

        # Draw edges with width based on weight
        edges = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges.keys(),
            width=[weight * 0.2 for weight in edges.values()],  # Scale edge thickness
            alpha=0.6
        )

        # Draw labels with larger font size
        nx.draw_networkx_labels(
            G, pos,
            font_size=10,
            font_color="black"
        )

        # Add title and save the plot
        plt.title("Author Collaboration Network", fontsize=16, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"../output/m2_authors/{filename}.svg", format="svg")
        plt.savefig(f"../output/m2_authors/{filename}.png", format="png")
        plt.show()

    @staticmethod
    def plot_histogram(df, column, title, filename, bins=10):
        """Plot a histogram for a given column."""

        # Plot the histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[column], bins=bins, color="#87CEEB", edgecolor="black", alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        plt.tight_layout()
        fig.savefig(f"../output/m2_authors/{filename}.svg", format="svg")
        fig.savefig(f"../output/m2_authors/{filename}.png", format="png")
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