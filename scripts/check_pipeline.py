#!/usr/bin/env python3
"""Check the cloud pipeline end to end with real credentials (reads .env; changes nothing).

    python scripts/check_pipeline.py                 # Azure OpenAI + Pinecone
    python scripts/check_pipeline.py --backend       # also call the deployed /api/query

Each check prints PASS/FAIL with the reason, so a broken setting is obvious before you debug code.
"""
import argparse
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

failures = 0
embedding_dim = None


def check(name, fn):
    global failures
    try:
        print(f"PASS  {name}: {fn()}")
    except Exception as e:
        failures += 1
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:300]}")


def embedding():
    from azure_embed import embed_texts

    global embedding_dim
    embedding_dim = len(embed_texts(["test"])[0])
    return f"{config.AZURE_OPENAI_EMBEDDING_MODEL} returns {embedding_dim}-dim vectors"


def pinecone_index():
    from pinecone import Pinecone

    stats = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX).describe_index_stats()
    msg = f"index '{config.PINECONE_INDEX}': dimension {stats.dimension}, {stats.total_vector_count} vectors"
    if embedding_dim and embedding_dim != stats.dimension:
        raise RuntimeError(f"embedding is {embedding_dim}-dim but {msg}")
    if stats.total_vector_count == 0:
        raise RuntimeError(f"{msg} (empty: run the daily job or backfill_staging.py)")
    return msg


def retrieval():
    from retrieve import retrieve

    hits = retrieve("How can businesses reduce the cost of running large language models?", k=3)
    if not hits:
        raise RuntimeError("no results")
    return "; ".join(f"{h['paper_id']} (distance {h['distance']})" for h in hits)


def blob_storage():
    from ingestion import _container

    names = [b.name for b in _container().list_blobs(results_per_page=1)][:1]
    return f"container '{config.PDF_CONTAINER}' reachable ({'has blobs' if names else 'empty'})"


def backend():
    base = config.BACKEND_API_URL
    if not base:
        raise RuntimeError("BACKEND_API_URL is not set")
    health = requests.get(f"{base}/api/health", timeout=120)
    resp = requests.post(f"{base}/api/query", json={"question": "What is model routing?", "k": 3},
                         headers={"x-functions-key": config.BACKEND_API_KEY}, timeout=180)
    if resp.status_code != 200:
        raise RuntimeError(f"/api/query returned {resp.status_code}: {resp.text[:200]}")
    body = resp.json()
    return f"health {health.status_code}, query ok: {len(body['sources'])} sources, timings {body['timings']}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", action="store_true", help="also test the deployed Function App")
    args = parser.parse_args()
    check("Azure OpenAI embeddings", embedding)
    check("Pinecone index", pinecone_index)
    check("Retrieval", retrieval)
    check("Blob Storage", blob_storage)
    if args.backend:
        check("Deployed backend", backend)
    sys.exit(1 if failures else 0)
