'''
This is EXACTLY Your Real Problem
------------------------------------
Correct info is indirect    ↓ similarity
Wrong chunks have keywords  ↑ similarity
Top-k = small               ✅ correct chunk dropped

Input_1 works as this matches but in Input_2 the correct answer is more indirect and the strong competitors 
have higher similarity, so the correct answer gets ranked lower and is dropped when top-k=2. 
In Input_3, the correct answer is even more indirect and gets ranked even lower, while the strong competitors and 
generic matches get ranked higher, leading to the correct answer being dropped from the top-k results.

Input_1:
documents = [
    "Dogs are loyal animals",   # ✅ correct answer
    "Dogs are commonly kept as pets",  # ⚠️ similar but weaker
    "Wolves are known for loyalty in packs",  # ⚠️ semantically close competitor
    "Cats love sleeping",
    "Loyalty is a valued trait in animals"  # ⚠️ broad semantic match
]

Output_1:
{'document': 'Dogs are loyal animals', 'score': np.float64(0.7621138389975138)}
{'document': 'Loyalty is a valued trait in animals', 'score': np.float64(0.740980491629257)}
{'document': 'Wolves are known for loyalty in packs', 'score': np.float64(0.6523768751970799)}
{'document': 'Dogs are commonly kept as pets', 'score': np.float64(0.38917259176601465)}
{'document': 'Cats love sleeping', 'score': np.float64(0.29377112678646966)}

Input_2:
documents = [
    "Dogs are often considered man's best companion due to their behavior",  # ✅ correct but indirect
    "Wolves are known for strong loyalty within their packs",               # ❌ strong competitor
    "Loyalty is an important behavioral trait observed in many animals",    # ❌ generic strong
    "Dogs are commonly kept as pets",                                       # weak mention
    "Cats love sleeping"
]

Output_2:
{'document': 'Loyalty is an important behavioral trait observed in many animals', 'score': np.float64(0.6988132123519631)}
{'document': 'Wolves are known for strong loyalty within their packs', 'score': np.float64(0.6344945745610387)}
{'document': "Dogs are often considered man's best companion due to their behavior", 'score': np.float64(0.49077741104076295)}
{'document': 'Dogs are commonly kept as pets', 'score': np.float64(0.38917259176601465)}
{'document': 'Cats love sleeping', 'score': np.float64(0.29377112678646966)}

Input_3:
documents = [
    "Dogs are widely kept as pets due to their companionship",
    "Wolves exhibit loyalty and coordination in hunting groups",
    "Animal behavior studies often analyze loyalty patterns",
    "Some animals form strong social bonds in groups",
    "Cats are independent animals"
]

Output_3:
{'document': 'Animal behavior studies often analyze loyalty patterns', 'score': np.float64(0.6642908394512498)}
{'document': 'Wolves exhibit loyalty and coordination in hunting groups', 'score': np.float64(0.5379766087303672)}
{'document': 'Some animals form strong social bonds in groups', 'score': np.float64(0.5252785905984657)}
{'document': 'Dogs are widely kept as pets due to their companionship', 'score': np.float64(0.4295674014910013)}
{'document': 'Cats are independent animals', 'score': np.float64(0.40663937580703563)}

'''

from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    "Dogs are loyal animals",   # ✅ correct answer
    "Dogs are commonly kept as pets",  # ⚠️ similar but weaker
    "Wolves are known for loyalty in packs",  # ⚠️ semantically close competitor
    "Cats love sleeping",
    "Loyalty is a valued trait in animals"  # ⚠️ broad semantic match
]

# documents = [
#     "Dogs are often considered man's best companion due to their behavior",  # ✅ correct but indirect
#     "Wolves are known for strong loyalty within their packs",               # ❌ strong competitor
#     "Loyalty is an important behavioral trait observed in many animals",    # ❌ generic strong
#     "Dogs are commonly kept as pets",                                       # weak mention
#     "Cats love sleeping"
# ]

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

# Query
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

results.sort(key=lambda x: x["score"], reverse=True)

print("\nTop Results:")
for r in results:
    print(r)

top_k = 2

print(f"\nTop {top_k} Results:")
for r in results[:top_k]:
    print(r)


