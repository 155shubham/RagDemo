from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

documents = [
    "Dogs are loyal animals",
    "Cats love sleeping",
    "SQL Server stores relational data",
    "Azure provides cloud services"
]

vectors = []

for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )

    vectors.append(response.data[0].embedding)

# query = "Tell me about databases"
# query = "Tell me cloud platforms"
query = "Which animal is loyal?"

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query
).data[0].embedding


def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)

    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))


results = []

for i, vec in enumerate(vectors):
    score = cosine_similarity(query_embedding, vec)
    results.append({
        "document": documents[i],
        "score": score
    })

results.sort(
    key=lambda x: x["score"],
    reverse=True
)

print("Best Match:")
print(results[0]["document"])
print(results[0]["score"])


# Notes
'''
Congratulations 🎉

You have now manually implemented:

User Query
    ↓
Embedding
    ↓
Cosine Similarity
    ↓
Top K Documents

This is the core retrieval step inside:

LangChain RetrievalQA
RAG
ChromaDB
Pinecone
Weaviate
Azure AI Search Vector Search

The only difference is that real vector databases do this over millions of vectors using optimized nearest-neighbor algorithms instead of looping through a Python list.
'''
