import chromadb

import config

BATCH_SIZE = 500


def get_collection():
    client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
    return client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_chunks(chunks: list[dict], embeddings: list[list[float]]):
    collection = get_collection()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=embeddings[i : i + BATCH_SIZE],
            documents=[c["chunk_text"] for c in batch],
            metadatas=[
                {"paper_id": c["paper_id"], "section_label": c["section_label"]}
                for c in batch
            ],
        )
    return collection
