"""
Multi-agent business-insight workflow (Roadmap Task 5).

    Retriever -> Analyst -> Reviewer (Analyst retries once with feedback) -> Business-insight agent

Only a reviewer-validated answer reaches the insight agent, and the insight is checked in code so
every claim cites a real source number. Fails closed: no validation, no insight.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict

import config
from generate import generate_answer
from retrieve import retrieve
from reviewer_agent import review_answer

MAX_ANALYST_RETRIES = 1   # retry once with the reviewer's feedback

# Cutoff on the top-1 distance. 0.48 fits today's bge embeddings (worst relevant 0.464, closest
# off-topic 0.485). Roadmap numbers for Azure text-embedding-3-small are 0.61 vs 0.73, so set
# NO_EVIDENCE_DISTANCE=0.67 in .env once Task 1 is merged.
NO_EVIDENCE_DISTANCE = float(os.getenv("NO_EVIDENCE_DISTANCE", "0.48"))

MATURITY = ("research", "early adoption", "production-ready")
CONFIDENCE = ("low", "medium", "high")
CITED_FIELDS = ("summary", "business_relevance", "opportunities", "risks")
CITE_RE = re.compile(r"\[(\d+)\]")


INSIGHT_PROMPT = """You are a business analyst at a consulting firm.
Turn the validated research answer below into a clear, evidence-based business insight.

RULES:
- Use ONLY the numbered sources and the validated answer. Do not introduce outside facts, market sizes, or company news.
- Cite every factual claim in "summary", "business_relevance", "opportunities", and "risks" with [n] markers that exist in the sources.
- Each citation must directly support the specific claim it accompanies. Do not attach citations merely because a source discusses a related topic.
- Preserve numerical values and their context accurately. Never combine results from different experiments as if they came from one.
- Use clear, natural language, complete sentences, and proper spacing. Avoid run-on sentences.
- Do not repeat the same finding unnecessarily.
- "industries" must include only industries explicitly supported by the research. Do not guess. Return [] if none are established.
- "recommended_actions" are suggestions, not research findings. Phrase them as things to consider, investigate, or test. Cite supporting evidence when applicable.
- "maturity" describes the maturity of the research or technology, not the maturity of the business opportunity.
- Set "confidence" based on evidence:
  - "high": multiple relevant sources consistently support the main conclusions.
  - "medium": evidence supports the main conclusions but has limited coverage or some uncertainty.
  - "low": evidence is sparse, indirect, or insufficient to support important conclusions.
- If the evidence is thin, explicitly acknowledge the limitation and set confidence to "low".
- Never claim that a result is production-ready unless the research directly establishes that.

SOURCES:
{sources}

VALIDATED ANSWER:
{answer}

QUESTION: {question}

Return JSON only, with exactly these keys:
{{"summary": str, "business_relevance": str, "industries": [str], "opportunities": [str],
"risks": [str], "maturity": "research" | "early adoption" | "production-ready",
"recommended_actions": [str], "confidence": "low" | "medium" | "high"}}"""


@dataclass
class InsightResult:
    question: str
    status: str = ""   # pending_review | insufficient_evidence | failed_validation | insight_failed
    answer: str = ""
    review: dict = field(default_factory=dict)
    insight: dict | None = None
    sources: list = field(default_factory=list)
    revisions: int = 0
    timings: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


# ---------- helpers ----------

_client = None


def _gemini():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _client


def _call_llm(prompt: str) -> str:
    from google.genai import types
    resp = _gemini().models.generate_content(
        model=config.GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json"),
    )
    if not resp.text:
        raise ValueError("Gemini returned an empty response.")
    return resp.text


def _distance(hit: dict):
    return hit.get("score", hit.get("distance"))


def _source(hit: dict) -> dict:
    keys = ("chunk_id", "paper_id", "title", "authors", "published", "pdf_url",
            "section_path", "page_start", "page_end", "text")
    return {**{k: hit.get(k) for k in keys}, "score": _distance(hit)}


def _format_sources(hits: list[dict]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        start, end = h.get("page_start"), h.get("page_end")
        pages = f"p.{start}" if start == end else f"p.{start}-{end}"
        blocks.append(f"[{i}] {h.get('title')} - {h.get('section_path')} ({pages})\n{h.get('text')}")
    return "\n\n".join(blocks)


def _tick(res: InsightResult, stage: str, t0: float) -> None:
    res.timings[stage] = round(res.timings.get(stage, 0.0) + time.perf_counter() - t0, 2)


def validate_insight(raw: str, n_sources: int) -> dict:
    """Parse the insight JSON and check structure, enums, and that every claim cites a real source."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Insight must be a JSON object.")

    for key in ("summary", "business_relevance"):
        if not isinstance(data.get(key), str) or not data[key].strip():
            raise ValueError(f"'{key}' must be a non-empty string.")
    for key in ("industries", "opportunities", "risks", "recommended_actions"):
        if not isinstance(data.get(key), list) or not all(isinstance(x, str) for x in data[key]):
            raise ValueError(f"'{key}' must be a list of strings.")
    if data.get("maturity") not in MATURITY:
        raise ValueError(f"'maturity' must be one of {MATURITY}.")
    if data.get("confidence") not in CONFIDENCE:
        raise ValueError(f"'confidence' must be one of {CONFIDENCE}.")

    for key in CITED_FIELDS:
        items = data[key] if isinstance(data[key], list) else [data[key]]
        for item in items:
            marks = {int(m) for m in CITE_RE.findall(item)}
            if not marks:
                raise ValueError(f"'{key}' has a claim with no [n] citation: {item[:80]!r}")
            if any(not 1 <= m <= n_sources for m in marks):
                raise ValueError(f"'{key}' cites a source that doesn't exist: {item[:80]!r}")
    for item in data["recommended_actions"]:
        if any(not 1 <= int(m) <= n_sources for m in CITE_RE.findall(item)):
            raise ValueError(f"'recommended_actions' cites a source that doesn't exist: {item[:80]!r}")
    return data


# ---------- agents ----------

def retriever_agent(question: str, k: int = 5):
    hits = retrieve(question, k=k)
    return hits, (_distance(hits[0]) if hits else None)


def analyst_agent(question: str, hits: list[dict], feedback: list[str] | None = None) -> str:
    prompt_q = question
    if feedback:
        prompt_q = (f"{question}\n\nA reviewer found problems with the previous draft. Fix them and "
                    "use only the excerpts:\n- " + "\n- ".join(feedback))
    return generate_answer(prompt_q, hits)


def reviewer_agent(hits: list[dict], answer: str) -> dict:
    return review_answer(hits, answer)


def insight_agent(question: str, answer: str, hits: list[dict]) -> dict:
    prompt = INSIGHT_PROMPT.format(sources=_format_sources(hits), answer=answer, question=question)
    last_error = None
    for _ in range(2):   # first try plus one repair attempt
        p = prompt if last_error is None else (
            prompt + f"\n\nYour previous output was rejected: {last_error}\nReturn corrected JSON only.")
        try:
            return validate_insight(_call_llm(p), len(hits))
        except ValueError as e:   # includes json.JSONDecodeError
            last_error = str(e)
    raise ValueError(f"Insight failed validation twice: {last_error}")


# ---------- orchestrator ----------

def run_pipeline(question: str, k: int = 5, mode: str = "insight") -> InsightResult:
    res = InsightResult(question=question)

    t0 = time.perf_counter()
    hits, best = retriever_agent(question, k)
    _tick(res, "retrieve", t0)
    res.trace.append(f"Retriever: {len(hits)} chunks, best distance {best}")
    if best is None or best > NO_EVIDENCE_DISTANCE:
        res.status = "insufficient_evidence"
        res.answer = "No relevant research found in the corpus for this question."
        res.trace.append(f"Best distance is above the cutoff {NO_EVIDENCE_DISTANCE}; stopped before generating.")
        return res
    res.sources = [_source(h) for h in hits]

    feedback = None
    for attempt in range(MAX_ANALYST_RETRIES + 1):
        res.revisions = attempt

        t0 = time.perf_counter()
        draft = analyst_agent(question, hits, feedback)
        _tick(res, "generate", t0)
        if not draft:
            res.status = "failed_validation"
            res.trace.append("Analyst: empty answer (API limit or error)")
            return res

        t0 = time.perf_counter()
        try:
            review = reviewer_agent(hits, draft)
        except Exception as e:   # fail closed: never release an unchecked answer
            res.status = "failed_validation"
            res.review = {"error": str(e)}
            res.trace.append(f"Reviewer: error ({e})")
            return res
        finally:
            _tick(res, "review", t0)

        res.review = review
        res.trace.append(f"Reviewer (draft {attempt + 1}): faithful={review['faithful']}, "
                         f"citations_valid={review['citations_valid']}, issues={len(review['issues'])}")
        if review["faithful"] and review["citations_valid"]:
            res.answer = draft
            break
        feedback = review["issues"] or ["Remove unsupported claims and fix invalid citations."]
    else:
        res.status = "failed_validation"
        res.trace.append(f"Still failing review after {MAX_ANALYST_RETRIES} retry; nothing released.")
        return res

    if mode != "insight":
        res.status = "pending_review"
        return res

    t0 = time.perf_counter()
    try:
        res.insight = insight_agent(question, res.answer, hits)
        res.status = "pending_review"
        res.trace.append("Insight agent: validated, every claim cites a real source")
    except Exception as e:
        res.status = "insight_failed"   # the validated answer is still returned
        res.trace.append(f"Insight agent: {e}")
    _tick(res, "insight", t0)
    return res


def run_insight(question: str, k: int = 5) -> InsightResult:
    return run_pipeline(question, k, mode="insight")


if __name__ == "__main__":
    flags = sys.argv[1:]
    mode = "answer" if "--answer" in flags else "insight"
    question = " ".join(a for a in flags if a != "--answer") or input("Question: ")
    out = asdict(run_pipeline(question, mode=mode))
    for s in out["sources"]:
        s.pop("text", None)   # keep the console output readable
    print(json.dumps(out, indent=2, ensure_ascii=False))