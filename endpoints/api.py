from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
from app.loader import Loader
from app.vector_store import Vector_Store
from app.rag_chain import Rag_Chain
from app.config import OPENAPI_API_KEY
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
txt_file_path = (Path(__file__).resolve().parent.parent /
                 "data" / "information.txt")

loader = Loader(path=txt_file_path)
documents = loader.document_load()

vec_store = Vector_Store()
vec_store.create_vector_store(documents)

rag = Rag_Chain(vs=vec_store)
rag.create_chain()


class QueryRequest(BaseModel):
    question: str


@app.post("/query")
async def query_rag(request: QueryRequest):
    # Here you would set up your RAG pipeline and run the query against it.
    # This is a placeholder implementation.
    # print(txt_file_path)
    answer = await rag.query(question=request.question)
    return {"answer": answer}

if __name__ == "__main__":
    print(txt_file_path)
