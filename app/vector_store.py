from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings


class Vector_Store:

    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vs = None
        pass

    def create_vector_store(self, documents: List[Document]):
        """Create vector store from chunks.
        Note:
            - Must return vector store.
        """
        self.vs = FAISS.from_documents(documents, self.embeddings)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search on vector store.
        Note:
            - Must return relevant documents.
        """
        if self.vs is None:
            raise ValueError("Vector store is not initialized.")

        relevant_docs = self.vs.similarity_search(query, k=k)
        return relevant_docs
