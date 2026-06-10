from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

text = "I love dogs"

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=text
)

embeddings = response.data[0].embedding

print(f"Dimensions: {len(embeddings)}")
print(f"Embeddings: {embeddings}")
