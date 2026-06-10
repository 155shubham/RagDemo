from openai import OpenAI
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

sentences = [
    "I love dogs",
    "Dogs are wondeerful pets",
    "Sql server is a database"
]

vectors = []

for sentence in sentences:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentence
    )
    embedding = response.data[0].embedding
    vectors.append(embedding)


def cosine_similarity(vec1, vec2):
    a = np.array(vec1)
    b = np.array(vec2)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


score1 = cosine_similarity(vectors[0], vectors[1])
score2 = cosine_similarity(vectors[0], vectors[2])

print("Dog vs Dogs:", score1)
print("Dog vs SQL Server:", score2)
