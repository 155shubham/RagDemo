import json

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "intro",
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Hybrid Search Demo - BM25 + Vector Search\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "data-ingestion",
            "metadata": {},
            "source": [
                "### Data Ingestion & Setup"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "load-data",
            "metadata": {},
            "outputs": [],
            "source": [
                "from pathlib import Path\n",
                "from langchain_community.document_loaders import TextLoader\n",
                "from langchain_text_splitters import RecursiveCharacterTextSplitter\n",
                "from langchain_openai import OpenAIEmbeddings\n",
                "from langchain_community.vectorstores import FAISS\n",
                "from dotenv import load_dotenv\n",
                "\n",
                "# Load file\n",
                "file_path = Path(\"../data/information.txt\").resolve()\n",
                "loader = TextLoader(file_path, encoding=\"utf-8\")\n",
                "documents = loader.load()\n",
                "print(f\"✓ Loaded {len(documents)} document(s)\")\n",
                "\n",
                "# Create chunks\n",
                "text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)\n",
                "chunks = text_splitter.split_documents(documents=documents)\n",
                "print(f\"✓ Created {len(chunks)} chunk(s)\")\n",
                "\n",
                "# Create embeddings\n",
                "load_dotenv()\n",
                "embedding_model = OpenAIEmbeddings(model=\"text-embedding-3-small\", dimensions=1024)\n",
                "\n",
                "# Create vector store\n",
                "vector_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)\n",
                "print(\"✓ Vector store created\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "hybrid-search-heading",
            "metadata": {},
            "source": [
                "### Hybrid Search (BM25 + Vector)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "hybrid-setup",
            "metadata": {},
            "outputs": [],
            "source": [
                "from langchain_community.retrievers import BM25Retriever\n",
                "from langchain.retrievers import EnsembleRetriever\n",
                "\n",
                "# Create BM25 retriever (keyword search)\n",
                "bm25_retriever = BM25Retriever.from_documents(chunks)\n",
                "\n",
                "# Create vector retriever (semantic search)\n",
                "vector_retriever = vector_store.as_retriever(search_kwargs={\"k\": 4})\n",
                "\n",
                "# Combine both retrievers\n",
                "hybrid_retriever = EnsembleRetriever(\n",
                "    retrievers=[bm25_retriever, vector_retriever],\n",
                "    weights=[0.5, 0.5]\n",
                ")\n",
                "\n",
                "print(\"✓ Hybrid retriever created (BM25 + Vector with equal weights)\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "hybrid-test",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Test hybrid search\n",
                "query = \"story of Captain Ahab\"\n",
                "results = hybrid_retriever.invoke(query)\n",
                "\n",
                "print(f\"Query: '{query}'\")\n",
                "print(f\"\\nFound {len(results)} relevant document(s):\\n\")\n",
                "\n",
                "for i, doc in enumerate(results, 1):\n",
                "    print(f\"{i}. {doc.page_content[:100]}...\\n\")"
            ]
        },
        {
            "cell_type": "markdown",
            "id": "weight-tuning",
            "metadata": {},
            "source": [
                "### Adjust Weights"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "id": "custom-weights",
            "metadata": {},
            "outputs": [],
            "source": [
                "# Try different weights\n",
                "custom_hybrid = EnsembleRetriever(\n",
                "    retrievers=[bm25_retriever, vector_retriever],\n",
                "    weights=[0.3, 0.7]  # More weight to semantic search\n",
                ")\n",
                "\n",
                "query = \"fantasy adventure\"\n",
                "results = custom_hybrid.invoke(query)\n",
                "\n",
                "print(f\"Query (weights 0.3 BM25, 0.7 Vector): '{query}'\")\n",
                "print(f\"Results: {len(results)} document(s)\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "RagDemo (3.11.15)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "name": "python",
            "version": "3.11.15"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open('experiment/hybrid_search.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Updated hybrid_search.ipynb")
