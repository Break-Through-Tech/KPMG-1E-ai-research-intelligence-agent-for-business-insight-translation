"""Evaluate answer quality on the benchmark question set.

For each in-scope question: retrieve top-k chunks, generate a business answer,
then judge it with Gemini on three dimensions plus a mechanical citation check:

- Faithfulness + citation validity: reviewer_agent.review_answer (existing judge).
- Answer relevance vs. the benchmark ground truth (1-5 rubric).
- Business relevance / output quality (1-5 rubric).
- Citation coverage: share of sentences carrying a valid [n] marker.

Writes per-question results to eval/results/generation_results.jsonl and
aggregate metrics to eval/results/generation_metrics.json.

    python eval/evaluate_generation.py
    python eval/evaluate_generation.py --limit 3
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import config  # noqa: E402
import generate  # noqa: E402
import retrieve as retrieval  # noqa: E402
import reviewer_agent  # noqa: E402

BENCHMARK_PATH = EVAL_DIR / "benchmark_questions.json"
RESULTS_DIR = EVAL_DIR / "results"

RELEVANCE_PROMPT = """You are scoring an AI-generated answer for relevance, not generating one.

QUESTION:
{question}

REFERENCE ANSWER (background material written from the source papers; the generated
answer may add or omit details and still be relevant — use this to check it addresses
the right substance):
{ground_truth}

GENERATED ANSWER TO SCORE:
{answer}

Score 1-5 how well the generated answer addresses the question:
1 = Does not address the question at all.
2 = Mostly off-topic; a few tangential points.
3 = Partially addresses the question but misses or misstates key substance.
4 = Addresses the question well; minor gaps.
5 = Fully addresses the question with the key substance.

Return JSON only: {{"score": <1-5>, "rationale": "<one sentence>"}}
"""

BUSINESS_PROMPT = """You are scoring an AI-generated answer for business usefulness, not generating one.

The system's goal: translate AI research into insights a business audience can act on.

QUESTION:
{question}

GENERATED ANSWER TO SCORE:
{answer}

Score 1-5 on business relevance and output quality:
1 = Technical jargon; unusable by a business reader.
2 = Understandable but no actionable insight.
3 = Some useful insight but buried in detail or incomplete.
4 = Clear, actionable insight for a business reader.
5 = Clear, actionable insight with a crisp executive takeaway; ready to forward to leadership.

Return JSON only: {{"score": <1-5>, "rationale": "<one sentence>"}}
"""


def _gemini_call(prompt: str, retries: int = 3) -> str:
    """One Gemini call with retry + rate-limit backoff (shared judge plumbing)."""
    from google.genai import errors, types

    client = reviewer_agent._gemini()
    gen_config = types.GenerateContentConfig(temperature=0.0)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=gen_config,
            )
            return (resp.text or "").strip()
        except errors.ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                if attempt == retries - 1:
                    raise
                time.sleep(10)
                continue
            raise
        except errors.ServerError:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _parse_score(text: str) -> dict:
    """Parse a {"score": int, "rationale": str} judge response."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    result = json.loads(cleaned)
    if not isinstance(result.get("score"), int) or not 1 <= result["score"] <= 5:
        raise ValueError(f"Judge returned an invalid score: {result}")
    return result


def citation_stats(answer: str, n_sources: int) -> dict:
    """Mechanical [n] marker check: validity and per-sentence coverage."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", answer.strip()) if s]
    markers = re.findall(r"\[(\d+)\]", answer)
    valid = [int(m) for m in markers if 1 <= int(m) <= n_sources]
    invalid = [int(m) for m in markers if not 1 <= int(m) <= n_sources]
    cited_sentences = sum(
        1 for s in sentences if re.search(r"\[\d+\]", s)
    )
    return {
        "n_markers": len(markers),
        "n_valid_markers": len(valid),
        "n_invalid_markers": len(invalid),
        "invalid_markers": invalid,
        "n_sentences": len(sentences),
        "n_cited_sentences": cited_sentences,
        "citation_coverage": round(cited_sentences / len(sentences), 3) if sentences else 0.0,
    }


def evaluate_question(q: dict, k: int) -> dict:
    question = q["query"]
    hits = retrieval.retrieve(question, k=k, use_hyde=False)
    answer = generate.generate_answer(question, hits)

    record = {
        "id": q["id"],
        "query": question,
        "category": q["category"],
        "k": k,
        "sources": [
            {
                "rank": i + 1,
                "chunk_id": h["chunk_id"],
                "paper_id": h["paper_id"],
                "section_path": h.get("section_path"),
            }
            for i, h in enumerate(hits)
        ],
        "answer": answer,
        "citation_stats": citation_stats(answer, len(hits)),
    }

    if not answer:
        record["skipped"] = "no answer generated (GEMINI_API_KEY missing or generation failed)"
        return record

    # Judge 1: faithfulness + citation validity against the retrieved sources.
    record["review"] = reviewer_agent.review_answer(hits, answer)
    time.sleep(4)

    # Judge 2: answer relevance vs. ground truth.
    relevance = _parse_score(
        _gemini_call(
            RELEVANCE_PROMPT.format(
                question=question,
                ground_truth=q.get("ground_truth_answer") or "(no reference answer available)",
                answer=answer,
            )
        )
    )
    record["relevance"] = relevance
    time.sleep(4)

    # Judge 3: business relevance / output quality.
    record["business_quality"] = _parse_score(
        _gemini_call(BUSINESS_PROMPT.format(question=question, answer=answer))
    )

    return record


def aggregate(records: list[dict]) -> dict:
    judged = [r for r in records if "review" in r]
    mean = statistics.mean

    def pct(values):
        return round(100 * mean(values), 1) if values else None

    return {
        "n_questions": len(records),
        "n_answered": len(judged),
        "n_skipped": len(records) - len(judged),
        "faithful_pct": pct([r["review"]["faithful"] for r in judged]),
        "citations_valid_pct": pct([r["review"]["citations_valid"] for r in judged]),
        "mean_answer_relevance_1_5": round(mean([r["relevance"]["score"] for r in judged]), 2) if judged else None,
        "mean_business_quality_1_5": round(mean([r["business_quality"]["score"] for r in judged]), 2) if judged else None,
        "mean_citation_coverage": round(mean([r["citation_stats"]["citation_coverage"] for r in judged]), 3) if judged else None,
        "answers_with_invalid_markers": sum(
            1 for r in judged if r["citation_stats"]["n_invalid_markers"] > 0
        ),
        "reviewer_issues_total": sum(len(r["review"]["issues"]) for r in judged),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, help="Only evaluate the first N in-scope questions")
    args = ap.parse_args()

    if not config.GEMINI_API_KEY:
        sys.exit("evaluate_generation.py needs GEMINI_API_KEY (answer generation + judging).")

    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = [q for q in json.load(f)["questions"] if not q["off_topic"]]
    if args.limit:
        questions = questions[: args.limit]

    records = []
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['id']}: {q['query'][:70]}", flush=True)
        try:
            records.append(evaluate_question(q, k=args.k))
        except Exception as e:
            records.append({"id": q["id"], "query": q["query"], "error": str(e)})
        if i < len(questions):
            time.sleep(4)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "generation_results.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    metrics = aggregate(records)
    with open(RESULTS_DIR / "generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nGeneration evaluation ({metrics['n_answered']}/{metrics['n_questions']} answered)")
    for key, value in metrics.items():
        print(f"  {key:32} {value}")
    print(f"\nWrote {RESULTS_DIR / 'generation_results.jsonl'}")
    print(f"Wrote {RESULTS_DIR / 'generation_metrics.json'}")


if __name__ == "__main__":
    main()
