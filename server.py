from typing import Dict, List, Union

from fastapi import FastAPI, HTTPException
import torch

from InstructorEmbedding import INSTRUCTOR

app = FastAPI()

instructor = INSTRUCTOR("hkunlp/instructor-large", device="cuda" if torch.cuda.is_available() else "cpu")

INSTRUCTION = "Represent the sentence for retrieving a duplicate sentence:"
BATCH_SIZE = 32


@app.post("/embed/")
def embed(texts_to_embed: List[str]):
    if not texts_to_embed:
        raise HTTPException(status_code=400, detail="Empty batch")

    batched_embeddings = [
        instructor.encode(
            [[INSTRUCTION, s] for s in texts_to_embed[i : i + BATCH_SIZE]],
            normalize_embeddings=True
        ) for i in range(0, len(texts_to_embed), BATCH_SIZE)
    ]

    return [embedding.tolist() for embeddings in batched_embeddings for embedding in embeddings]
