
'''
Simple LLM-based Reranker (reranking using LLM)

'''

from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# documents = [
#     "Dogs are loyal animals",   # ✅ correct answer
#     "Dogs are commonly kept as pets",  # ⚠️ similar but weaker
#     "Wolves are known for loyalty in packs",  # ⚠️ semantically close competitor
#     "Cats love sleeping",
#     "Loyalty is a valued trait in animals"  # ⚠️ broad semantic match
# ]

documents = [
    "Dogs are often considered man's best companion due to their behavior",  # ✅ correct but indirect
    "Wolves are known for strong loyalty within their packs",               # ❌ strong competitor
    "Loyalty is an important behavioral trait observed in many animals",    # ❌ generic strong
    "Dogs are commonly kept as pets",                                       # weak mention
    "Cats love sleeping"
]

# documents = [
#     "Dogs are widely kept as pets due to their companionship",
#     "Wolves exhibit loyalty and coordination in hunting groups",
#     "Animal behavior studies often analyze loyalty patterns",
#     "Some animals form strong social bonds in groups",
#     "Cats are independent animals"
# ]


vectors = []

# Generate embeddings
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    vectors.append(response.data[0].embedding)

def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b)/(np.linalg.norm(a) * np.linalg.norm(b))

queries = [
    "Which animal is loyal?",
    "Which pet is loyal?",
    "Dogs loyalty behavior"
]

all_results = []

for q in queries:
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=q
    ).data[0].embedding

    for i, vec in enumerate(vectors):
        score = cosine_similarity(query_embedding, vec)
        all_results.append({
            "query": q,
            "document": documents[i],
            "score": score
        })

# Sort globally
all_results.sort(key=lambda x: x["score"], reverse=True)

# Print top results
print("\nTop Results Across All Queries:")
for r in all_results[:5]:
    print(r)


prompt_1 = """
    You are a system that selects the most relevant document.

    Query:
    {query}

    Documents:
    {formatted_docs}

    Return ONLY the document number that best answers the query.
"""

prompt_2 = """
    You are selecting the best answer.

    Query:
    {query}

    Documents:
    {formatted_docs}

    Important:
    - Consider implicit meaning (e.g., "best companion" implies loyalty)
    - Choose the BEST answer even if it is indirect

    Return ONLY the document number.
"""

def rerank_with_llm(query, candidates):
    formatted_docs = "\n\n".join(
        [f"{i+1}. {doc}" for i, doc in enumerate(candidates)]
    )

    prompt = prompt_2.format(query=query, formatted_docs=formatted_docs)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    choice = int(response.choices[0].message.content.strip())

    return candidates[choice - 1]


TOP_K = 5

candidates = [r["document"] for r in all_results[:TOP_K]]

best_doc = rerank_with_llm(
    "Which animal is loyal?",
    candidates
)

print("\nBest after reranking:")
print(best_doc)




