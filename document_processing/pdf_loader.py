from langchain_community.document_loaders import PyMuPDFLoader
from preprocessing.cleaning import clean_text
import os


def load_papers(file_paths):

    all_documents = []

    for file_path in file_paths:

        loader = PyMuPDFLoader(file_path)

        documents = loader.load()

        paper_name = os.path.basename(file_path)

        for doc in documents:

            # Clean extracted page text
            doc.page_content = clean_text(
                doc.page_content
            )

            # Preserve metadata
            doc.metadata["paper_name"] = paper_name
            doc.metadata["source"] = file_path

        # Avoid storing empty pages
        documents = [
            doc for doc in documents
            if doc.page_content.strip()
        ]

        all_documents.extend(documents)

    return all_documents