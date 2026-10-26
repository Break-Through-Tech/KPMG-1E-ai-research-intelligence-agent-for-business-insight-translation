"""Render reports/evaluation_report.md from the JSON results in eval/results/.

Run after evaluate_retrieval.py / evaluate_generation.py (directly or via
run_eval.py). Reads whatever result files exist, so a retrieval-only run still
produces a report with the generation section marked as skipped.

    python eval/make_report.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

RESULTS_DIR = EVAL_DIR / "results"
REPORT_PATH = EVAL_DIR.parent / "reports" / "evaluation_report.md"


def load_json(name: str) -> dict | None:
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(name: str) -> list[dict]:
    path = RESULTS_DIR / name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def retrieval_section(retrieval: dict | None, per_question: list[dict]) -> list[str]:
    lines = ["## Retrieval relevance", ""]
    if not retrieval:
        return lines + [
            "_No retrieval results found — run `python eval/evaluate_retrieval.py`._",
            "",
        ]

    lines += [
        f"Relevance rule: a hit is relevant when it comes from the expected paper "
        f"AND its section_path starts with one of the expected section prefixes "
        f"(k={retrieval['k']}, {retrieval['n_questions']} in-scope questions, "
        f"{retrieval['n_off_topic']} off-topic probes).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Paper Hit@1 | {fmt(retrieval['paper Hit@1'])} |",
        f"| Paper Precision@k | {fmt(retrieval['paper Precision@k'])} |",
        f"| Section Hit@1 | {fmt(retrieval['section Hit@1'])} |",
        f"| Section Hit@k | {fmt(retrieval['section Hit@k'])} |",
        f"| Section MRR | {fmt(retrieval['section MRR'])} |",
        f"| Section Recall@k | {fmt(retrieval['section Recall@k'])} |",
        f"| Retrieval misses@k | {retrieval['section retrieval misses@k']} |",
        f"| Mean relevant distance@k | {fmt(retrieval['mean relevant distance@k'])} |",
        f"| Worst relevant distance@k | {fmt(retrieval['worst relevant distance@k'])} |",
        f"| Closest off-topic distance | {fmt(retrieval['closest off-topic distance'])} |",
        f"| Seconds per query | {fmt(retrieval['seconds per query'])} |",
        "",
    ]

    # Per-category breakdown.
    categories: dict[str, list[dict]] = {}
    for record in per_question:
        categories.setdefault(record["category"], []).append(record)
    if categories:
        lines += [
            "Per-category breakdown:",
            "",
            "| Category | n | Paper Hit@1 | Section Hit@k | MRR |",
            "|---|---|---|---|---|",
        ]
        for cat in sorted(categories):
            rows = categories[cat]
            n = len(rows)
            ph1 = sum(r["paper_hit1"] for r in rows) / n
            sh = sum(r["section_hit"] for r in rows) / n
            mrr = sum(r["reciprocal_rank"] for r in rows) / n
            lines.append(
                f"| {cat} | {n} | {ph1:.3f} | {sh:.3f} | {mrr:.3f} |"
            )
        lines.append("")

    misses = [r for r in per_question if not r["section_hit"]]
    if misses:
        lines += ["Questions with no relevant section in the top k:", ""]
        for r in misses:
            lines.append(f"- **{r['id']}** — {r['query']}")
        lines.append("")
    return lines


def generation_section(generation: dict | None, per_question: list[dict]) -> list[str]:
    lines = ["## Generation quality", ""]
    if not generation:
        return lines + [
            "_No generation results found — run `python eval/evaluate_generation.py` "
            "(requires GEMINI_API_KEY)._",
            "",
        ]

    lines += [
        f"{generation['n_answered']} of {generation['n_questions']} questions answered and judged "
        f"({generation['n_skipped']} skipped).",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Faithful answers | {fmt(generation['faithful_pct'], 1)}% |",
        f"| Citations valid (reviewer) | {fmt(generation['citations_valid_pct'], 1)}% |",
        f"| Mean answer relevance (1-5) | {fmt(generation['mean_answer_relevance_1_5'], 2)} |",
        f"| Mean business quality (1-5) | {fmt(generation['mean_business_quality_1_5'], 2)} |",
        f"| Mean citation coverage | {fmt(generation['mean_citation_coverage'])} |",
        f"| Answers with invalid [n] markers | {generation['answers_with_invalid_markers']} |",
        f"| Reviewer issues raised | {generation['reviewer_issues_total']} |",
        "",
    ]

    # Low scores and reviewer flags.
    flagged = [
        r for r in per_question
        if "review" in r and (
            not r["review"]["faithful"]
            or not r["review"]["citations_valid"]
            or r["citation_stats"]["n_invalid_markers"] > 0
            or r.get("relevance", {}).get("score", 5) <= 2
            or r.get("business_quality", {}).get("score", 5) <= 2
        )
    ]
    if flagged:
        lines += ["Flagged answers (unfaithful, invalid citations, or score <= 2):", ""]
        for r in flagged:
            notes = []
            if not r["review"]["faithful"]:
                notes.append("unfaithful")
            if not r["review"]["citations_valid"]:
                notes.append("invalid citations (reviewer)")
            if r["citation_stats"]["n_invalid_markers"] > 0:
                notes.append(f"invalid markers {r['citation_stats']['invalid_markers']}")
            if r.get("relevance", {}).get("score", 5) <= 2:
                notes.append(f"relevance {r['relevance']['score']}/5")
            if r.get("business_quality", {}).get("score", 5) <= 2:
                notes.append(f"business {r['business_quality']['score']}/5")
            lines.append(f"- **{r['id']}** — {', '.join(notes)} — {r['query']}")
        lines.append("")

    errors = [r for r in per_question if "error" in r]
    if errors:
        lines += ["Errors:", ""]
        for r in errors:
            lines.append(f"- **{r['id']}** — {r['error']}")
        lines.append("")
    return lines


def per_question_table(per_question: list[dict]) -> list[str]:
    if not per_question:
        return []
    lines = [
        "## Per-question results",
        "",
        "| ID | Relevance 1-5 | Business 1-5 | Faithful | Citations valid | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for r in per_question:
        if "review" not in r:
            lines.append(f"| {r['id']} | — | — | — | — | — |")
            continue
        lines.append(
            f"| {r['id']} "
            f"| {r.get('relevance', {}).get('score', '—')} "
            f"| {r.get('business_quality', {}).get('score', '—')} "
            f"| {'✓' if r['review']['faithful'] else '✗'} "
            f"| {'✓' if r['review']['citations_valid'] else '✗'} "
            f"| {fmt(r['citation_stats']['citation_coverage'])} |"
        )
    lines.append("")
    return lines


def main():
    retrieval = load_json("retrieval_metrics.json")
    generation = load_json("generation_metrics.json")
    retrieval_q = load_jsonl("retrieval_results.jsonl")
    generation_q = load_jsonl("generation_results.jsonl")

    REPORT_PATH.parent.mkdir(exist_ok=True)

    lines = [
        "# Evaluation Report — AI Research Intelligence Agent",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Benchmark: `eval/benchmark_questions.json` — 20 in-scope business questions over "
        "the 5-paper corpus plus 4 off-topic probes. Retrieval = bge-base-en-v1.5 + ChromaDB "
        "(cosine); answers = Gemini with [n]-cited business-insight prompt; judging = Gemini "
        "(faithfulness/citations via reviewer_agent, relevance and business quality via 1-5 rubrics).",
        "",
        "## Summary",
        "",
    ]
    if retrieval:
        lines += [
            f"- **Retrieval:** paper Hit@1 {fmt(retrieval['paper Hit@1'])}, "
            f"section Hit@k {fmt(retrieval['section Hit@k'])}, "
            f"MRR {fmt(retrieval['section MRR'])}, "
            f"{retrieval['section retrieval misses@k']} miss(es) on {retrieval['n_questions']} questions.",
        ]
    if generation:
        lines += [
            f"- **Generation:** {fmt(generation['faithful_pct'], 1)}% faithful, "
            f"{fmt(generation['citations_valid_pct'], 1)}% valid citations, "
            f"relevance {fmt(generation['mean_answer_relevance_1_5'], 2)}/5, "
            f"business quality {fmt(generation['mean_business_quality_1_5'], 2)}/5.",
        ]
    if not retrieval and not generation:
        lines.append("- No results found. Run `python eval/run_eval.py` first.")
    lines.append("")

    lines += retrieval_section(retrieval, retrieval_q)
    lines += generation_section(generation, generation_q)
    lines += per_question_table(generation_q)
    lines += findings_section()

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


def findings_section() -> list[str]:
    """Curated findings from eval/findings.md, so re-running the report keeps them."""
    findings_path = EVAL_DIR / "findings.md"
    lines = ["## Findings and improvement opportunities", ""]
    if findings_path.exists():
        lines += [
            finding.rstrip() for finding in findings_path.read_text(encoding="utf-8").splitlines()
        ] + [""]
    else:
        lines += [
            "_(Curate this section in `eval/findings.md` — it is included here on every "
            "report render.)_",
            "",
        ]
    return lines


if __name__ == "__main__":
    main()
