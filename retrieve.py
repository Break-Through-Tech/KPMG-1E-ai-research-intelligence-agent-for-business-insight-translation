import time

import numpy as np

import config
from azure_embed import embed_texts

HYDE_PROMPT = """Write a short passage (about 120 words) from an AI research paper that \
directly answers the question below. Use the technical terminology a researcher would use \
(methods, metrics, findings), not business language. Output only the passage.

Question: {question}"""

_client = None
_index = None
_hyde_cache: dict[str, str] = {}
_hyde_disabled_reason = ""


def _gemini():
    global _client
    if _client is None:
        from google import genai

        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _pinecone_index():
    global _index
    if _index is None:
        from pinecone import Pinecone

        _index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
    return _index


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
    # No query prefix: the bge instruction doesn't apply to text-embedding-3-small, and the
    # indexed chunks (see ingestion.index_chunks) are embedded as plain text.
    texts = [question]
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

    `where` is a metadata filter such as {"paper_id": {"$in": [...]}}.

    Each result includes both "distance" and "score" set to the same cosine-distance
    value (lower = more similar) — kept as two keys temporarily so eval_retrieval.py,
    reviewer_agent.py, and the notebook's generation code don't all need to be patched
    at once. Pick one name as a team and drop the other once everything's migrated.
    """
    result = _pinecone_index().query(
        vector=query_embedding,
        top_k=k,
        filter=where,
        include_metadata=True,
        namespace=config.PINECONE_NAMESPACE,
    )
    hits = []
    for match in result.matches:
        meta = match.metadata or {}
        # Pinecone returns cosine similarity; downstream code expects distance (lower = closer).
        dist_val = round(1 - float(match.score), 4)
        hits.append({
            "chunk_id": match.id,
            "paper_id": meta.get("paper_id"),
            "title": meta.get("title"),
            "authors": meta.get("authors"),
            "published": meta.get("published"),
            "pdf_url": meta.get("pdf_url"),
            "section_label": meta.get("section_label"),
            "section_path": meta.get("section_path"),
            # Pinecone stores every number as a float.
            "page_start": int(meta.get("page_start", 0)),
            "page_end": int(meta.get("page_end", 0)),
            "text": meta.get("text", ""),
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