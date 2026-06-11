"""Text embedding functions."""
from __future__ import annotations


def get_embedder():
    """Lazy-loaded sentence transformer model. Returns callables embed(text) -> list[float] and embed_many(texts) -> list[list[float]]."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed(text: str) -> list[float]:
        return model.encode(text, normalize_embeddings=True).tolist()

    def embed_many(texts: list[str]) -> list[list[float]]:
        return model.encode(texts, normalize_embeddings=True).tolist()

    return embed, embed_many