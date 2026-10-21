import time

import numpy as np

import config
import store
from embed import embed_texts

HYDE_PROMPT = """Write a short passage (about 120 words) from an AI research paper that \
directly answers the question below. Use the technical terminology a researcher would use \
(methods, metrics, findings), not business language. Output only the passage.

Question: {question}"""

_client = None
_hyde_cache: dict[str, str] = {}
_hyde_disabled_reason = ""


def _gemini():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def hypothetical_document(question: str, retries: int = 3) -> str:
    """HyDE: an LLM-written passage that looks like the chunks we want to find.
    Returns "" when no API key is set or the call fails, so retrieval still works."""
    global _hyde_disabled_reason
    if not config.GEMINI_API_KEY or _hyde_disabled_reason:
        return ""
    if question in _hyde_cache:
        return _hyde_cache[question]
    from google.genai import errors, types

    gen_config = types.GenerateContentConfig(
        temperature=0.3,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    for attempt in range(retries):
        try:
            resp = _gemini().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=HYDE_PROMPT.format(question=question),
                config=gen_config,
            )
            doc = (resp.text or "").strip()
            if doc:
                _hyde_cache[question] = doc
            return doc
        except errors.ServerError as e:
            if attempt == retries - 1:
                print(f"HyDE generation failed, using the raw question only: {e}")
                return ""
            time.sleep(2**attempt)
        except errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                if attempt == retries - 1:
                    print(
                        f"HyDE rate-limited, using the raw question only this time: {e}"
                    )
                    return ""

                time.sleep(5)  # Allow headroom for the free tier's 15 req/min limit.
                continue

            _hyde_disabled_reason = str(e)
            print(
                f"HyDE disabled for this session, using the raw question only: {e}"
            )
            return ""

        except Exception as e:
            _hyde_disabled_reason = str(e)
            print(f"HyDE disabled for this session, using the raw question only: {e}")
            return ""


def embed_query(question: str, use_hyde: bool = False) -> list[float]:
    """Average the question and HyDE passage embeddings, then re-normalize for cosine search."""
    texts = [config.QUERY_INSTRUCTION + question]
    if use_hyde and (doc := hypothetical_document(question)):
        texts.append(doc)
    vec = np.mean(embed_texts(texts), axis=0)
    norm = np.linalg.norm(vec)

    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Query embedding has an invalid or zero norm.")
    return (vec / norm).tolist()


def search(
    query_embedding: list[float],
    k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Retrieval layer: return the k chunks closest to an already-embedded query,
    best first.

    Each result includes both "distance" and "score" set to the same cosine-distance
    value (lower = more similar) — kept as two keys temporarily so eval_retrieval.py,
    reviewer_agent.py, and the notebook's generation code don't all need to be patched
    at once. Pick one name as a team and drop the other once everything's migrated.
    """
    collection = store.get_collection()
    store.check_embedding_model(collection)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": k,
    }

    if where:
        query_kwargs["where"] = where

    result = collection.query(**query_kwargs)

    hits = []
    for chunk_id, doc, meta, dist in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        dist_val = round(float(dist), 4)
        hits.append({
            "chunk_id": chunk_id,
            "paper_id": meta.get("paper_id"),
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "published": meta.get("published"),
            "pdf_url": meta.get("pdf_url"),
            "section_label": meta.get("section_label"),
            "section_path": meta.get("section_path"),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "text": doc,
            "distance": dist_val,
            "score": dist_val,
        })
    return hits


def retrieve(
    question: str,
    k: int = 5,
    use_hyde: bool = False,
    where: dict | None = None,
) -> list[dict]:
    """Embed (and optionally enrich) the question, then search. Same output as search()."""
    return search(embed_query(question, use_hyde), k=k, where=where)