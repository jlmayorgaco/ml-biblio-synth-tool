# ML Zotero Biblio Tool

![Zotero](https://img.shields.io/badge/Zotero-Integration-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-NLP-green)
![License](https://img.shields.io/badge/License-Apache%202.0-yellowgreen)

## Overview

**ML Zotero Biblio Tool** is designed to automate bibliometric analysis by integrating with Zotero. Using advanced Machine Learning (ML) techniques and Natural Language Processing (NLP), this tool helps researchers efficiently analyze large collections of academic documents. It provides insights into trends, key topics, and relationships between documents.

### Key Features

- **Zotero Integration:** Seamlessly connects to your Zotero library to import bibliographic data and PDFs.
- **Topic Modeling:** Uses Latent Dirichlet Allocation (LDA) to discover underlying themes in your document collection.
- **Trend Analysis:** Identifies emerging research trends through keyword frequency and topic shifts.
- **Document Classification:** Automatically classifies documents using supervised machine learning models.
- **Clustering:** Groups similar documents with unsupervised clustering algorithms (K-means, DBSCAN).
- **Co-citation Networks:** Visualizes relationships between papers based on citation patterns.
- **Automatic Report Generation:** Exports analysis results in PDF and HTML formats, with visualizations.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Zotero API Key (for integration)
- Required libraries:
  - `PyZotero`
  - `pandas`
  - `matplotlib`
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

3. Set up your Zotero API Key:
   - Go to [Zotero Settings](https://www.zotero.org/settings/keys) and generate a new API key.
   - Export your Zotero library to a `BibTeX` file or integrate directly via the API.

### Usage

1. Run the main script to analyze your Zotero collection:
    ```bash
    python analyze.py --bibtex path_to_your_bibtex_file.bib --output report.pdf
    ```

2. To use the Zotero API integration:
    ```bash
    python analyze.py --zotero-api-key your_api_key --output report.pdf
    ```

### Example

```bash
python analyze.py --bibtex my_zotero_library.bib --output trends_analysis.html
