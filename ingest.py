import argparse
import json
import sys

import config

REQUIRED_FIELDS = ("chunk_id", "paper_id", "section_label", "chunk_text")


def load_chunks(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            if path.endswith(".jsonl"):  # chunking notebook writes one JSON object per line
                chunks = [json.loads(line) for line in f if line.strip()]
            else:
                chunks = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"Error: could not read chunks file '{path}': {e}")
    if not isinstance(chunks, list):
        sys.exit(f"Error: '{path}' must contain a JSON list of chunk objects.")
    seen = set()
    for i, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            sys.exit(f"Error: chunk at index {i} is not an object.")
        missing = [f for f in REQUIRED_FIELDS if f not in chunk]
        if missing:
            sys.exit(f"Error: chunk at index {i} is missing field(s): {', '.join(missing)}")
        if chunk["chunk_id"] in seen:
            sys.exit(f"Error: duplicate chunk_id '{chunk['chunk_id']}' at index {i}.")
        seen.add(chunk["chunk_id"])
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Embed chunks and store them in ChromaDB.")
    parser.add_argument("--chunks", default="./chunks.json")
    parser.add_argument("--db-path", help="Override CHROMA_DB_PATH")
    parser.add_argument("--model", help="Override EMBEDDING_MODEL")
    args = parser.parse_args()

    # Overrides must be applied before embed/store read config.
    if args.db_path:
        config.CHROMA_DB_PATH = args.db_path
    if args.model:
        config.EMBEDDING_MODEL = args.model

    chunks = load_chunks(args.chunks)
    if not chunks:
        sys.exit("Error: no chunks to ingest.")

    from embed import embed_texts
    from store import ingest_chunks

    embeddings = embed_texts([c["chunk_text"] for c in chunks])
    collection = ingest_chunks(chunks, embeddings)
    print(f"Collection '{collection.name}' now contains {collection.count()} chunks.")


if __name__ == "__main__":
    main()
