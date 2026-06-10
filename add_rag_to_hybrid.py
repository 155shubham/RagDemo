import json

with open('experiment/hybrid_search.ipynb', 'r') as f:
    nb = json.load(f)

# Add RAG chain cell
rag_chain_cell = {
    "cell_type": "markdown",
    "id": "rag-chain-heading",
    "metadata": {},
    "source": [
        "### Use Hybrid Results in RAG Chain"
    ]
}

rag_invoke_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "rag-with-hybrid",
    "metadata": {},
    "outputs": [],
    "source": [
        "from langchain_core.prompts import ChatPromptTemplate\n",
        "from langchain_openai import ChatOpenAI\n",
        "from langchain_core.output_parsers import StrOutputParser\n",
        "\n",
        "# Format documents for context\n",
        "def format_docs(docs):\n",
        "    return \"\\n\\n\".join([doc.page_content for doc in docs])\n",
        "\n",
        "# Create RAG chain\n",
        "prompt = ChatPromptTemplate.from_messages([\n",
        "    (\"system\", \"Answer based on this context:\\n{context}\"),\n",
        "    (\"user\", \"{question}\")\n",
        "])\n",
        "\n",
        "llm = ChatOpenAI(model=\"gpt-3.5-turbo\", temperature=0)\n",
        "output_parser = StrOutputParser()\n",
        "\n",
        "rag_chain = prompt | llm | output_parser\n",
        "\n",
        "# Get hybrid search results and pass to RAG chain\n",
        "query = \"What is Moby Dick about?\"\n",
        "context_docs = hybrid_retriever.invoke(query)\n",
        "context = format_docs(context_docs)\n",
        "\n",
        "answer = rag_chain.invoke({\n",
        "    \"context\": context,\n",
        "    \"question\": query\n",
        "})\n",
        "\n",
        "print(f\"Question: {query}\\n\")\n",
        "print(f\"Answer: {answer}\")"
    ]
}

nb['cells'].append(rag_chain_cell)
nb['cells'].append(rag_invoke_cell)

with open('experiment/hybrid_search.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✓ Added RAG chain cell that uses hybrid search results as context")
