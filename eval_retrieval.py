"""Compare baseline (raw question) vs HyDE retrieval on a small labeled question set.

A result counts as relevant when it comes from the expected paper and its section_path
starts with one of the listed sections. Off-topic questions have no relevant chunks; they
show whether a distance cutoff could separate "nothing relevant" from real matches.

    python eval_retrieval.py --runs 3
"""
import argparse
import statistics
import time

import retrieve as r

K = 5

# (question, expected paper_id, relevant section_path prefixes)
QUESTIONS = [
    ("We use several AI vendors; how do we decide which model should handle each customer request without overspending?",
     "2608.20316", ["Abstract", "1 Introduction", "2 Setting", "4 Pandora", "7 Conclusion"]),
    ("Is it worth paying extra to check how good a model's answer will be before choosing it?",
     "2608.20316", ["Abstract", "1 Introduction", "2 Setting", "3 The Pandora", "4 Pandora"]),
    ("Could independent AI providers bid on tasks in a marketplace?",
     "2608.20316", ["Abstract", "5 Pandora", "D Supplemental Results > D.2"]),
    ("How can businesses reduce LLM costs?",
     "2608.20316", ["Abstract", "1 Introduction", "2 Setting", "4 Pandora", "7 Conclusion"]),
    ("Can AI systems improve how future AI models are trained?",
     "2608.20318", ["Abstract", "1 Introduction", "6 Conclusion"]),
    ("How reliable are AI agents at designing new algorithms?",
     "2608.20318", ["Abstract", "1 Introduction", "3 Experiments", "4 Analysis", "6 Conclusion"]),
    ("Does letting an AI agent think longer lead to better research results?",
     "2608.20318", ["4 Analysis > 4.2", "3 Experiments"]),
    ("How should we benchmark an AI coding agent's ability to do real ML research?",
     "2608.20318", ["Abstract", "1 Introduction", "2 AI4AI-Bench"]),
    ("Can AI agents learn workflows by watching employees use software?",
     "2608.20319", ["Abstract", "1 Introduction", "3 Method", "7 Conclusion"]),
    ("How can we document how employees actually do their jobs from screen recordings for process auditing?",
     "2608.20319", ["Abstract", "1 Introduction", "2 Problem Formulation", "3 Method"]),
    ("Employees juggle several tasks at once; can AI separate their interleaved activities?",
     "2608.20319", ["Abstract", "1 Introduction", "3 Method > 3.2", "4 Intrinsic Evaluation"]),
    ("What are the privacy risks of recording employee computer activity?",
     "2608.20319", ["Ethics Statement", "Limitations"]),
    ("Can chatbots replace traditional customer surveys for collecting data?",
     "2608.20320", ["Abstract", "2 Background and Related Work > 2.2", "3 Methodology > 3.3", "4 Case Study > 4.1"]),
    ("How does bad weather change how people commute?",
     "2608.20320", ["Abstract", "5 Results > 5.1", "6 Discussion", "7 Conclusions"]),
    ("Can off-the-shelf LLMs predict customer choices as well as traditional machine learning?",
     "2608.20320", ["Abstract", "5 Results > 5.2", "5 Results > 5.3", "6 Discussion", "7 Conclusions"]),
    ("How do locally deployed LLMs compare with traditional statistical and machine-learning models for travel-mode prediction?",
     "2608.20320", ["Abstract", "4 Case Study > 4.3", "5 Results > 5.2", "6 Discussion", "7 Conclusions"]),
    ("How can AI help hospitals communicate with patients?",
     "2608.20331", ["Abstract", "1 Introduction", "5 Conclusion"]),
    ("How do we stop a healthcare chatbot from making up medical facts?",
     "2608.20331", ["Abstract", "1 Introduction", "3 Methods > 3.2"]),
    ("How can we train an AI assistant to tailor explanations to each user's needs?",
     "2608.20331", ["Abstract", "3 Methods > 3.3", "3 Methods > 3.4"]),
    ("Which parts of the reward design matter most for medical report explanations?",
     "2608.20331", ["4 Experiments > 4.3"]),
]

OFF_TOPIC = [
    "What is the best marketing strategy for retail stores during the holidays?",
    "How do interest rate changes affect mortgage lending?",
    "What are best practices for onboarding new employees?",
    "How do I set up a Kubernetes cluster?",
]


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
