# src/data_processing.py

import bibtexparser
import pdfplumber
import re
import os

class DataLoader:
    """Class for loading and preprocessing data from BibTeX and PDF files."""
    
    def __init__(self, bib_file_path, pdf_folder_path):
        self.bib_file_path = bib_file_path
        self.pdf_folder_path = pdf_folder_path

    def load_bib_file(self):
        """Loads and parses the BibTeX file."""
        with open(self.bib_file_path) as bibtex_file:
            bib_database = bibtexparser.load(bibtex_file)
        return bib_database.entries

    def match_references_with_pdfs(self, references):
        """Matches BibTeX references with available PDF files based on folder ID and reports unmatched references and files."""
        matched = []
        unmatched_references = []
        unmatched_files = []
        
        # Get the list of PDF folders
        pdf_folders = [f for f in os.listdir(self.pdf_folder_path) if os.path.isdir(os.path.join(self.pdf_folder_path, f))]
        
        # Iterate through each reference to match with a PDF in the specified folder
        for ref in references:
            file_field = ref.get('file', '')
            
            if 'files/' in file_field:
                folder_id = file_field.split('files/')[1].split('/')[0]
                folder_path = os.path.join(self.pdf_folder_path, folder_id)
                
                if os.path.exists(folder_path):
                    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
                    
                    if pdf_files:
                        matched_pdf_path = os.path.join(folder_path, pdf_files[0])
                        matched.append((ref, matched_pdf_path))
                    else:
                        unmatched_references.append(ref)
                else:
                    unmatched_references.append(ref)
            else:
                unmatched_references.append(ref)
        
        # Identify unmatched PDF folders
        matched_folders = {ref.get('file', '').split('files/')[1].split('/')[0] for ref, _ in matched if 'files/' in ref.get('file', '')}
        unmatched_files = [folder for folder in pdf_folders if folder not in matched_folders]

        # Final report of unmatched references and files
        print("\nMatching process completed.")
        print(f"Total references matched: {len(matched)} out of {len(references)}")
        print(f"Unmatched References: {len(unmatched_references)}")
        for ref in unmatched_references:
            print(f"- Title: {ref.get('title', 'No Title')} | Author: {ref.get('author', 'Unknown')}")
        
        print(f"\nUnmatched PDF Folders: {len(unmatched_files)}")
        for folder in unmatched_files:
            print(f"- Folder ID: {folder}")

        return matched

    def extract_text_from_pdf(self, pdf_path):
        """Extracts text from a PDF with adjustments for character spacing and layout."""
        extracted_text = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    text = page.extract_text(x_tolerance=2, y_tolerance=1).split('\n')  # Adjust tolerances as needed
                    if text:
                        # Filter out non-printable characters for readability
                        filtered_text = ''.join(c if c.isprintable() else ' ' for c in text)
                        extracted_text.append(filtered_text)
                except Exception as e:
                    print(f"Skipping unreadable content on page {i+1} in {pdf_path}: {e}")
        
        return ' '.join(extracted_text)

    def preprocess_text(self, text):
        """Basic text preprocessing: removing special characters."""
        return re.sub(r'\W+', ' ', text).lower()

    def load_and_process(self):
        """Loads and processes data from BibTeX and PDF files, returning structured data."""
        references = self.load_bib_file()
        matched_files = self.match_references_with_pdfs(references)

        # Structured output in { bibliographic_metadata: ..., plain_text: ... } format
        processed_data = [
            {   
                "id": ref.get("ID", "unknown_id"),
                "bibliographic_metadata": ref,
                "plain_text": self.preprocess_text(self.extract_text_from_pdf(pdf))
            }
            for ref, pdf in matched_files
        ]

        return processed_data
