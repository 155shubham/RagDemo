import asyncio

from app.loader import Loader
from app.vector_store import Vector_Store
from app.rag_chain import Rag_Chain
from pathlib import Path
from app.config import OPENAPI_API_KEY


base_file_path = Path(__file__).resolve().parent
txt_file_path = base_file_path / "data" / "information.txt"


async def test_rag_pipeline():
    print("Testing RAG pipeline...")
    # Here you would set up your RAG pipeline and run tests against it.

    # print("txt_file_path:", txt_file_path)
    doc_loader = Loader(path=txt_file_path)
    documents = doc_loader.document_load()
    print(f"Loaded {len(documents)} document(s).")

    # Print first 200 characters of the first document
    # print("Sample document content:", documents[0].page_content[:200])

    vec_store = Vector_Store()
    vec_store.create_vector_store(documents)
    # query = "What does the document talk about?"
    query = "Give an overview of the document"
    relevant_docs = vec_store.similarity_search(query)
    # print(
    #     f"Found {len(relevant_docs)} relevant document(s) for the query: {query}")
    # print("Sample relevant document content:",
    #       relevant_docs[0].page_content[:200])

    rag = Rag_Chain(vs=vec_store)
    rag.create_chain()
    response = await rag.query(question=query)
    print("RAG chain response:", response)

    print("RAG pipeline test completed.")


if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())
