from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os


class Loader:
    def __init__(self, path):
        """Initialise file path."""
        self.path = path

    def document_load(self):
        """Load articles from data/books.txt Text file.
        Note:
            - Must return documents.
            - Raise FileNotFoundError if the file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError("File not found.")

        text_loader = TextLoader(self.path, encoding="utf-8")
        documents = text_loader.load()
        return documents

    def create_chunks(self, documents) -> List[Document]:
        """Split the documents into chunks of size 500 and overlap of 50.
        Returns the created chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        return chunks
