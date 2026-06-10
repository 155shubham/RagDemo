
'''
Multi-query expansion

queries = [
    "Which animal is loyal?",
    "Which pet is loyal?",
    "Dogs loyalty behavior"
]

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







