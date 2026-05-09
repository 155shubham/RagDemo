from typing import List
from langchain_core import Document
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings


class vector_store:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = None
        pass

    def create_vector_store(self, documents: List[Document]):
        """Create vector store from chunks.
        Note:
            - Must return vector store.
        """
        self.vector_store = FAISS.from_documents(documents, self.embeddings)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on vector store.
        Note:
            - Must return relevant documents.
        """
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized.")

        relevant_docs = self.vector_store.similarity_search(query, k=k)
        return relevant_docs
