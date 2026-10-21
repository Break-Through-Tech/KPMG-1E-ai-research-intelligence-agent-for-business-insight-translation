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
        except Exception as e:
            # Client errors (bad model name, quota exhausted, invalid key) won't fix themselves
            # on retry, so skip HyDE for the rest of the session instead of failing per query.
            _hyde_disabled_reason = str(e)
            print(f"HyDE disabled for this session, using the raw question only: {e}")
            return ""


def embed_query(question: str, use_hyde: bool = True) -> list[float]:
    """Average the question and HyDE passage embeddings, then re-normalize for cosine search."""
    texts = [config.QUERY_INSTRUCTION + question]
    if use_hyde and (doc := hypothetical_document(question)):
        texts.append(doc)
    vec = np.mean(embed_texts(texts), axis=0)
    return (vec / np.linalg.norm(vec)).tolist()


def search(query_embedding: list[float], k: int = 5, where: dict | None = None) -> list[dict]:
    """
    Retrieval layer: return the k chunks closest to an already-embedded query, best first.
    ... (docstring unchanged) ...
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

    return [
        {
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
            "score": round(dist, 4),
        }
        for chunk_id, doc, meta, dist in zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        )
    ]


def retrieve(question: str, k: int = 5, use_hyde: bool = True, where: dict | None = None) -> list[dict]:
    """Embed (and optionally enrich) the question, then search. Same output as search()."""
    return search(embed_query(question, use_hyde), k=k, where=where)
