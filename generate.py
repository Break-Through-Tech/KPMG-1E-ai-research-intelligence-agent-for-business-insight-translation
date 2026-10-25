import time

import config

ANSWER_PROMPT = """You are a research analyst translating AI research papers into business insights.

Answer the question using ONLY the numbered excerpts from research papers below.

Rules:
- Write for a business audience: plain language, short paragraphs, explain technical terms in passing.
- Cite every claim with the excerpt it came from, like [1] or [2][3].
- If the excerpts do not answer the question, say so plainly. Do not use outside knowledge or invent findings.
- End with a one-line "Bottom line:" takeaway a busy executive could quote.

Excerpts:
{context}

Question: {question}"""

_client = None
_gen_disabled_reason = ""


def _gemini():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def format_source(i: int, hit: dict) -> str:
    pages = f"p.{hit['page_start']}" + (f"-{hit['page_end']}" if hit["page_end"] != hit["page_start"] else "")
    return f"[{i}] {hit['title']} - {hit['section_path']} ({pages})"


def build_context(hits: list[dict]) -> str:
    return "\n\n".join(f"{format_source(i, h)}\n{h['text']}" for i, h in enumerate(hits, 1))


def generate_answer(question: str, hits: list[dict], retries: int = 3) -> str:
    global _gen_disabled_reason
    if not hits:
        return "No matching chunks were found in the corpus for this question."
    if not config.GEMINI_API_KEY or _gen_disabled_reason:
        return ""
    from google.genai import errors, types

    gen_config = types.GenerateContentConfig(temperature=0.2)
    prompt = ANSWER_PROMPT.format(context=build_context(hits), question=question)
    for attempt in range(retries):
        try:
            resp = _gemini().models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config,
            )
            return (resp.text or "").strip()
        except errors.ServerError as e:
            if attempt == retries - 1:
                print(f"Answer generation failed after {retries} attempts: {e}")
                return ""
            time.sleep(2**attempt)
        except Exception as e:
            _gen_disabled_reason = str(e)
            print(f"Answer generation disabled for this session: {e}")
            return ""