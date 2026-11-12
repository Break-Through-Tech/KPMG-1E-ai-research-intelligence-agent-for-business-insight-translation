"""Embeddings from the Azure OpenAI text-embedding-3-small deployment (1536 dimensions).

Used by the cloud ingestion job; the Pinecone index is built from these vectors, so queries
must be embedded the same way. The standard OpenAI client talks to Azure's "/openai/v1" API
directly (the model name is the deployment name).
"""
from urllib.parse import urlparse

import config

BATCH_SIZE = 64

_client = None


def _openai():
    global _client
    if _client is None:
        from openai import OpenAI

        if not (config.AZURE_OPENAI_ENDPOINT and config.AZURE_OPENAI_KEY):
            raise RuntimeError("AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY must be set.")
        # Accept both the portal's bare resource URL and the full ".../openai/v1" form.
        host = urlparse(config.AZURE_OPENAI_ENDPOINT).netloc
        _client = OpenAI(base_url=f"https://{host}/openai/v1/", api_key=config.AZURE_OPENAI_KEY)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        resp = _openai().embeddings.create(
            model=config.AZURE_OPENAI_EMBEDDING_MODEL, input=texts[i : i + BATCH_SIZE]
        )
        vectors.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
    return vectors
