from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class HybridSearch:
    """Hybrid search combining BM25 (keyword) + Vector search (semantic)."""

    def __init__(self, vector_store, weight_bm25=0.5, weight_vector=0.5):
        """
        Args:
            vector_store: FAISS or other vector store instance
            weight_bm25: Weight for BM25 keyword search (0-1)
            weight_vector: Weight for vector semantic search (0-1)
        """
        self.vector_store = vector_store
        self.weight_bm25 = weight_bm25
        self.weight_vector = weight_vector
        self.bm25_retriever = None
        self.ensemble_retriever = None

    def setup(self, documents: List[Document]):
        """Initialize retrievers from documents."""
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 4})

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[self.weight_bm25, self.weight_vector]
        )

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of relevant documents
        """
        if self.ensemble_retriever is None:
            raise ValueError(
                "Hybrid search not initialized. Call setup() first.")

        results = self.ensemble_retriever.invoke(query)
        return results[:k]
