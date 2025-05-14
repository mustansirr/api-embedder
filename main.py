from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()
model = SentenceTransformer("all-MiniLM-L6-v2")

class EmbedRequest(BaseModel):
    input: str

@app.post("/embed")
async def embed(request: EmbedRequest):
    embedding = model.encode([request.input])[0]
    return {"embedding": embedding.tolist()}
