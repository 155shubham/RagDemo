from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter, Semantic
from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os


class DocLoader:
    def __init__(self, path):
        """Initialise file path."""
        self.path = path

    def plain_text_load(self) -> List[Document]:
        """Load articles from a text file.
        Note:
            - Must return documents.
            - Raise FileNotFoundError if the file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError("File not found.")

        text_loader = TextLoader(self.path, encoding="utf-8")
        documents = text_loader.load()
        return documents

    def pdf_load(self) -> List[Document]:
        """Load a PDF file and return documents.
        Note:
            - Must return documents.
            - Raise FileNotFoundError if the file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError("File not found.")

        pdf_loader = PyPDFLoader(self.path)
        documents = pdf_loader.load()
        return documents

    def create_chunks(self, documents) -> List[Document]:
        """Split the documents into chunks of size 1000 and overlap of 200.
        Returns the created chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        return chunks
    
    def create_chunks(self, documents) -> List[Document]:
        splitter = Seman
