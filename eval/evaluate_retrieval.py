"""Evaluate retrieval quality on the benchmark question set.

A result counts as relevant when it comes from the expected paper and its
section_path starts with one of the listed section prefixes (same rule as the
root eval_retrieval.py). Off-topic questions have no relevant chunks; they show
whether a distance cutoff could separate "nothing relevant" from real matches.

Writes per-question results to eval/results/retrieval_results.jsonl and
aggregate metrics to eval/results/retrieval_metrics.json.

    python eval/evaluate_retrieval.py
    python eval/evaluate_retrieval.py --k 5 --limit 3
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import retrieve as r  # noqa: E402

BENCHMARK_PATH = EVAL_DIR / "benchmark_questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_benchmark() -> list[dict]:
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        return json.load(f)["questions"]


def is_relevant(hit: dict, paper: str, sections: list[str]) -> bool:
    return hit["paper_id"] == paper and any(
        (hit.get("section_path") or "").startswith(s) for s in sections
    )


def evaluate(questions: list[dict], k: int) -> dict:
    in_scope = [q for q in questions if not q["off_topic"]]
    off_topic = [q for q in questions if q["off_topic"]]

    per_question = []
    paper_hit1, paper_p5, sec_hit1, sec_hit5, mrr = [], [], [], [], []
    sec_recall, relevant_distances, secs = [], [], []

    for q in in_scope:
        t = time.perf_counter()
        hits = r.retrieve(q["query"], k=k, use_hyde=False)
        elapsed = time.perf_counter() - t
        secs.append(elapsed)

        rel = [is_relevant(h, q["expected_paper_id"], q["expected_section_prefixes"]) for h in hits]
        paper_hits = sum(h["paper_id"] == q["expected_paper_id"] for h in hits)
        n_expected = max(len(q["expected_section_prefixes"]), 1)

        record = {
            "id": q["id"],
            "query": q["query"],
            "category": q["category"],
            "hits": [
                {
                    "rank": i + 1,
                    "chunk_id": h["chunk_id"],
                    "paper_id": h["paper_id"],
                    "section_path": h.get("section_path"),
                    "distance": h["distance"],
                    "relevant": relevant,
                }
                for i, (h, relevant) in enumerate(zip(hits, rel))
            ],
            "paper_hit1": bool(hits) and hits[0]["paper_id"] == q["expected_paper_id"],
            "paper_hits_in_top_k": paper_hits,
            "section_hit": any(rel),
            "section_hit1": bool(rel) and rel[0],
            "reciprocal_rank": next((1 / (i + 1) for i, x in enumerate(rel) if x), 0.0),
            "section_recall_at_k": min(sum(rel) / n_expected, 1.0),
            "seconds": round(elapsed, 3),
        }
        per_question.append(record)

        paper_hit1.append(record["paper_hit1"])
        paper_p5.append(paper_hits / k)
        sec_hit1.append(record["section_hit1"])
        sec_hit5.append(record["section_hit"])
        mrr.append(record["reciprocal_rank"])
        sec_recall.append(record["section_recall_at_k"])

        matching = [h["distance"] for h, relevant in zip(hits, rel) if relevant]
        if matching:
            relevant_distances.append(min(matching))

    off_topic_distances = []
    for q in off_topic:
        hits = r.retrieve(q["query"], k=1, use_hyde=False)
        if hits:
            off_topic_distances.append(hits[0]["distance"])

    mean = statistics.mean
    metrics = {
        "n_questions": len(in_scope),
        "k": k,
        "paper Hit@1": mean(paper_hit1) if paper_hit1 else 0.0,
        "paper Precision@k": mean(paper_p5) if paper_p5 else 0.0,
        "section Hit@1": mean(sec_hit1) if sec_hit1 else 0.0,
        "section Hit@k": mean(sec_hit5) if sec_hit5 else 0.0,
        "section MRR": mean(mrr) if mrr else 0.0,
        "section Recall@k": mean(sec_recall) if sec_recall else 0.0,
        "section retrieval misses@k": sum(1 for r_ in per_question if not r_["section_hit"]),
        "mean relevant distance@k": mean(relevant_distances) if relevant_distances else None,
        "worst relevant distance@k": max(relevant_distances) if relevant_distances else None,
        "n_off_topic": len(off_topic),
        "closest off-topic distance": min(off_topic_distances) if off_topic_distances else None,
        "seconds per query": mean(secs) if secs else 0.0,
    }

    return {"metrics": metrics, "per_question": per_question}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, help="Only evaluate the first N in-scope questions")
    args = ap.parse_args()

    questions = load_benchmark()
    if args.limit:
        questions = [q for q in questions if q["off_topic"]][: args.limit] + [
            q for q in questions if not q["off_topic"]
        ][: args.limit]

    r.retrieve("warm up the embedding model", k=1, use_hyde=False)
    result = evaluate(questions, k=args.k)

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "retrieval_results.jsonl", "w", encoding="utf-8") as f:
        for record in result["per_question"]:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(RESULTS_DIR / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result["metrics"], f, indent=2)

    print(f"\nRetrieval evaluation ({result['metrics']['n_questions']} questions, k={args.k})")
    for key, value in result["metrics"].items():
        if isinstance(value, float):
            print(f"  {key:32} {value:8.3f}")
        else:
            print(f"  {key:32} {value}")
    print(f"\nWrote {RESULTS_DIR / 'retrieval_results.jsonl'}")
    print(f"Wrote {RESULTS_DIR / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
