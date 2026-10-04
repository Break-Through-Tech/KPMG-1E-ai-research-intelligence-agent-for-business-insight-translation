from sentence_transformers import SentenceTransformer

import config

_model = None


def get_model() -> SentenceTransformer:
    """Lazily create the single shared model used for indexing and queries."""
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = get_model().encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return vectors.tolist()
