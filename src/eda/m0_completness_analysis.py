# -------------------------------------------------------------- #
# -- EDA M0 :: Completness ------------------------------------- #
# -------------------------------------------------------------- #

import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter

# -------------------------------------------------------------- #


class CompletenessProcessor:
    """Handles data processing for completeness and missing fields."""

    def __init__(self, processed_data):
        self.processed_data = processed_data
        self.essential_fields = ['title', 'author', 'year', 'doi']

    def calculate_completeness(self):
        """Calculate completeness metrics for each field."""
        completeness_data = []
        for field in self.essential_fields:
            filled_count = sum(1 for entry in self.processed_data if entry['bibliographic_metadata'].get(field))
            completeness_percentage = (filled_count / len(self.processed_data)) * 100
            completeness_data.append({'Field': field, 'Filled': filled_count, 'Completeness (%)': completeness_percentage})
        return pd.DataFrame(completeness_data)

    def identify_missing_fields(self):
        """Identify records with missing fields."""
        missing_fields_summary = []
        for i, entry in enumerate(self.processed_data):
            missing_fields = [field for field in self.essential_fields if not entry['bibliographic_metadata'].get(field)]
            if missing_fields:
                missing_fields_summary.append({"Record": i + 1, "Missing Fields": missing_fields})
        return pd.DataFrame(missing_fields_summary)


class CompletenessVisualizer:
    """Handles visualization of completeness metrics."""

    @staticmethod
    def save_plot(fig, filename):
        """Save the plot as SVG and PNG."""
        fig.savefig(f"../output/m0_completness/{filename}.svg", format="svg")
        fig.savefig(f"../output/m0_completness/{filename}.png", format="png")
        #plt.close(fig)
        plt.show()  # Display the plot in the notebook

    @staticmethod
    def plot_completeness(completeness_df, filename="completeness_plot"):
        """Plot completeness metrics as bar and pie charts."""
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14, 6))

        # Bar Chart for Completeness Percentage by Field
        ax1.barh(completeness_df['Field'], completeness_df['Completeness (%)'], color='#a6cee3')
        ax1.set_xlabel('Completeness (%)', labelpad=10)
        ax1.set_title('Completeness of Fields', pad=12, fontweight='bold')
        ax1.set_xlim(0, 100)
        ax1.invert_yaxis()
        ax1.grid(True, which='major', linestyle='--', linewidth=0.5, color='gray')
        ax1.axvline(x=80, color='red', linestyle=':', linewidth=1.5, label='80% Threshold')
        ax1.legend(loc='lower right')

        # Pie Chart for Filled Counts by Field
        ax2.pie(completeness_df['Filled'], labels=completeness_df['Field'], autopct='%1.1f%%',
                startangle=140, colors=['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c'])
        ax2.set_title('Proportion of Non-Missing Fields', pad=15, fontweight='bold')

        plt.tight_layout()
        CompletenessVisualizer.save_plot(fig, filename)

    @staticmethod
    def plot_missing_fields(missing_fields_df, filename="missing_fields_plot"):
        """Plot missing fields metrics as bar and pie charts."""
        all_missing_fields = [field for fields in missing_fields_df['Missing Fields'] for field in fields]
        missing_field_counts = Counter(all_missing_fields)
        fields, counts = zip(*missing_field_counts.most_common())

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Bar plot for missing fields frequency
        ax1.bar(fields, counts, color='#87CEEB')
        ax1.set_title('Frequency of Missing Fields', pad=12, fontweight='bold')
        ax1.set_xlabel('Missing Field')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', linestyle='--', color='gray', linewidth=0.5)

        # Pie chart for missing fields proportion
        ax2.pie(counts, labels=fields, autopct='%1.1f%%', startangle=140,
                colors=['#87CEEB', '#FFA07A', '#FFD700'])
        ax2.set_title('Proportion of Missing Fields', pad=12, fontweight='bold')

        plt.tight_layout()
        CompletenessVisualizer.save_plot(fig, filename)


class CompletenessReporter:
    """Handles reporting of completeness and missing fields in tabular format."""

    @staticmethod
    def report_completeness(completeness_df, filename="completeness_report"):
        """Print and save completeness metrics as a table and CSV."""
        print("Completeness Metrics")
        print(completeness_df.to_string(index=False))
        completeness_df.to_csv(f"../output/m0_completness/{filename}.csv", index=False)

    @staticmethod
    def report_missing_fields(missing_fields_df, filename="missing_fields_report"):
        """Print and save missing fields summary as a table and CSV."""
        print("\nMissing Fields Summary")
        if missing_fields_df.empty:
            print("All records have complete data!")
        else:
            print(missing_fields_df.to_string(index=False))
            missing_fields_df.to_csv(f"../output/m0_completness/{filename}.csv", index=False)

