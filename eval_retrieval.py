"""Compare baseline (raw question) vs HyDE retrieval on a small labeled question set.

Questions live in eval/benchmark_questions.json (single source of truth, shared
with eval/evaluate_retrieval.py and eval/evaluate_generation.py).

A result counts as relevant when it comes from the expected paper and its section_path
starts with one of the listed sections. Off-topic questions have no relevant chunks; they
show whether a distance cutoff could separate "nothing relevant" from real matches.

    python eval_retrieval.py --runs 3
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import retrieve as r

K = 5

BENCHMARK_PATH = Path(__file__).resolve().parent / "eval" / "benchmark_questions.json"


def load_benchmark():
    """Return (in_scope tuples, off_topic questions) from the shared benchmark file."""
    with open(BENCHMARK_PATH, encoding="utf-8") as f:
        questions = json.load(f)["questions"]
    in_scope = [
        (q["query"], q["expected_paper_id"], q["expected_section_prefixes"])
        for q in questions
        if not q["off_topic"]
    ]
    off_topic = [q["query"] for q in questions if q["off_topic"]]
    return in_scope, off_topic


QUESTIONS, OFF_TOPIC = load_benchmark()


def is_relevant(hit, paper, sections):
    return hit["paper_id"] == paper and any(hit["section_path"].startswith(s) for s in sections)



def evaluate(use_hyde):
    paper_hit1, paper_p5 = [], []
    sec_hit1, sec_hit5, mrr = [], [], []
    relevant_distances, retrieval_misses = [], []
    best_off, secs = [], []

    for q, paper, sections in QUESTIONS:
        t = time.perf_counter()
        hits = r.retrieve(q, k=K, use_hyde=use_hyde)
        secs.append(time.perf_counter() - t)

        rel = [is_relevant(h, paper, sections) for h in hits]
        paper_hit1.append(bool(hits) and hits[0]["paper_id"] == paper)
        paper_p5.append(sum(h["paper_id"] == paper for h in hits) / K)
        sec_hit1.append(bool(rel) and rel[0])
        sec_hit5.append(any(rel))
        mrr.append(next((1 / (i + 1) for i, x in enumerate(rel) if x), 0.0))

        # Measure distance only among relevant chunks in the top K.
        matching_distances = [
            h["distance"] for h, relevant in zip(hits, rel) if relevant
        ]
        if matching_distances:
            relevant_distances.append(min(matching_distances))
        else:
            retrieval_misses.append(q)
            print(f"\nRETRIEVAL MISS (HyDE={use_hyde}): {q}")
            for rank, hit in enumerate(hits, start=1):
                print(
                    f"  {rank}. paper={hit.get('paper_id')} "
                    f"section={hit.get('section_path') or hit.get('section_label')} "
                    f"distance={hit.get('distance')}"
                )

    for q in OFF_TOPIC:
        hits = r.retrieve(q, k=1, use_hyde=use_hyde)
        if hits:
            best_off.append(hits[0]["distance"])

    mean = statistics.mean
    return {
        "HyDE passages generated": (
            len(r._hyde_cache) / (len(QUESTIONS) + len(OFF_TOPIC))
            if use_hyde else 0.0
        ),
        "paper Hit@1": mean(paper_hit1),
        "paper Precision@5": mean(paper_p5),
        "section Hit@1": mean(sec_hit1),
        "section Hit@5": mean(sec_hit5),
        "section MRR": mean(mrr),
        "section retrieval misses@5": len(retrieval_misses),
        "mean relevant distance@5": (
            mean(relevant_distances) if relevant_distances else float("nan")
        ),
        "worst relevant distance@5": (
            max(relevant_distances) if relevant_distances else float("nan")
        ),
        "closest off-topic distance": (
            min(best_off) if best_off else float("nan")
        ),
        "seconds per in-scope query": mean(secs),
    }



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="HyDE runs (passages vary between runs)")
    args = ap.parse_args()

    r.retrieve("warm up the embedding model", k=1, use_hyde=False)
    baseline = evaluate(use_hyde=False)
    hyde_runs = []
    for _ in range(args.runs):
        r._hyde_cache.clear()
        r._hyde_disabled_reason = ""
        hyde_runs.append(evaluate(use_hyde=True))

    print(f"\n{'metric':32} {'baseline':>9} {'HyDE mean':>10} {'HyDE min':>9} {'HyDE max':>9}")
    for m, b in baseline.items():
        vals = [run[m] for run in hyde_runs]
        print(f"{m:32} {b:9.3f} {statistics.mean(vals):10.3f} {min(vals):9.3f} {max(vals):9.3f}")


if __name__ == "__main__":
    main()
