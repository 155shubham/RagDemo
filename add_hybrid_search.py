import json

with open('experiment/rag.ipynb', 'r') as f:
    nb = json.load(f)

# Add hybrid search cell
hybrid_search_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'hybrid-search-cell',
    'metadata': {},
    'outputs': [],
    'source': [
        '# Hybrid Search: BM25 + Vector Search\n',
        'from langchain_community.retrievers import BM25Retriever\n',
        'from langchain.retrievers import EnsembleRetriever\n',
        '\n',
        '# Create BM25 retriever (keyword search)\n',
        'bm25_retriever = BM25Retriever.from_documents(chunks)\n',
        '\n',
        '# Create vector retriever\n',
        'vector_retriever = vector_store.as_retriever(search_kwargs={"k": 4})\n',
        '\n',
        '# Combine both retrievers (ensemble)\n',
        'ensemble_retriever = EnsembleRetriever(\n',
        '    retrievers=[bm25_retriever, vector_retriever],\n',
        '    weights=[0.5, 0.5]\n',
        ')\n',
        'print("✓ Hybrid retriever (BM25 + Vector)")'
    ]
}

# Add reranking cell
rerank_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'reranking-cell',
    'metadata': {},
    'outputs': [],
    'source': [
        '# Reranking with Cohere\n',
        'from langchain_cohere import CohereReranker\n',
        'from langchain.retrievers.contextual_compression import ContextualCompressionRetriever\n',
        '\n',
        'compressor = CohereReranker(model="rerank-english-v2.0")\n',
        'compression_retriever = ContextualCompressionRetriever(\n',
        '    base_compressor=compressor,\n',
        '    base_retriever=ensemble_retriever\n',
        ')\n',
        'print("✓ Reranker added (Cohere)")'
    ]
}

# Add hybrid search demo cell
hybrid_demo_cell = {
    'cell_type': 'code',
    'execution_count': None,
    'id': 'hybrid-demo-cell',
    'metadata': {},
    'outputs': [],
    'source': [
        '# Test Hybrid Search + Reranking\n',
        'query = "story of Captain Ahab"\n',
        'reranked_docs = compression_retriever.invoke(query)\n',
        'print(f"Retrieved and reranked {len(reranked_docs)} document(s)")\n',
        'for i, doc in enumerate(reranked_docs, 1):\n',
        '    print(f"\\n{i}. {doc.page_content[:80]}...")'
    ]
}

# Insert cells after vector store
nb['cells'].insert(10, hybrid_search_cell)
nb['cells'].insert(11, rerank_cell)
nb['cells'].insert(12, hybrid_demo_cell)

with open('experiment/rag.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Added hybrid search + reranking to notebook")
