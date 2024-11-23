# ML Zotero Biblio Tool

![Zotero](https://img.shields.io/badge/Zotero-Integration-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen)

## Overview

**ML Zotero Biblio Tool** is an advanced bibliometric analysis framework designed for researchers who want to analyze their custom Zotero collections. With a combination of Machine Learning (ML), Natural Language Processing (NLP), and sophisticated visualization tools, this project enables in-depth exploration of bibliographic datasets, providing actionable insights into trends, topics, author collaborations, and more.

This project was developed using a sample dataset related to power systems frequency estimation as a demonstration, but the framework is designed to be flexible and domain-independent.

---

## Key Features

- **Zotero Integration:**
  - Seamlessly integrates with your Zotero library to import bibliographic metadata and full-text PDFs.
  - Supports both manual import of `BibTeX` files and direct API connections with Zotero.

- **Document Processing and Cleaning:**
  - Extracts, tokenizes, and lemmatizes text from PDFs for analysis.
  - Handles metadata parsing, including authors, publication years, and journals.

- **Topic Modeling:**
  - Uses Latent Dirichlet Allocation (LDA) to uncover themes and topics within your document collection.

- **Author and Source Analysis:**
  - Identifies the most prolific authors, collaboration networks, and trends in publishing over time.
  - Highlights key journals and sources contributing to your research domain.

- **Trend and Citation Analysis:**
  - Tracks keyword trends and shifts in topics over time.
  - Generates co-citation networks to visualize relationships between documents.

- **Clustering and Classification:**
  - Groups similar documents using unsupervised algorithms like K-means and DBSCAN.
  - Automatically classifies documents with supervised ML models.

- **Automatic Report Generation:**
  - Exports analysis results, including visualizations, in both PDF and HTML formats.

---

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Zotero API Key (for integration)
- Required libraries:
  - `PyZotero`
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `nltk`, `spaCy`
  - `scikit-learn`, `gensim`
  - `WeasyPrint`, `pdfkit`

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ml-zotero-biblio-tool.git
    cd ml-zotero-biblio-tool
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Analyze your Zotero collection using the provided notebooks:
   - The repository includes a series of Jupyter Notebooks located in the `notebooks` directory. Each notebook corresponds to a specific step or type of analysis.
   - Launch Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open the desired notebook from the `notebooks` directory and follow the instructions provided in each cell to process your data.

4. Results:
   - All analysis outputs, including visualizations, tables, and reports, are saved automatically in the `output` directory. This ensures that you can easily access and review the results of your bibliometric analysis.


---

### Output

- **PDF Report:** A comprehensive report including all visualizations and analysis results.
- **HTML Report:** An interactive report for online sharing and exploration.
- **CSV Files:** Export of raw data and analysis results for further exploration.

---

## Roadmap

Future features planned for this tool include:
- Support for additional bibliographic formats (e.g., RIS, EndNote).
- Integration with external citation databases like Scopus or Web of Science.
- Enhanced visualization options with Plotly.
- Advanced ML features like word embeddings and semantic similarity analysis.

---

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

## Contributing

We welcome contributions! Please fork the repository and submit a pull request for any new features or bug fixes.
