from embed import embed_texts
from store import get_collection


def retrieve(question: str, k: int = 5) -> list[dict]:
    """
    Returns the k most similar chunks as:
    [{"paper_id": str, "section_label": str, "text": str, "score": float}, ...]
    Lower score = more similar (cosine distance).
    """
    collection = get_collection()
    result = collection.query(
        query_embeddings=embed_texts([question]),
        n_results=k,
    )
    return [
        {
            "paper_id": meta["paper_id"],
            "section_label": meta["section_label"],
            "text": doc,
            "score": round(dist, 4),
        }
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]
