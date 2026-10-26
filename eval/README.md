# Evaluation framework

Measures retrieval quality, answer faithfulness, citation correctness, and business
usefulness of the RAG pipeline over the paper corpus in `chunks/` (ChromaDB).

## Benchmark

`benchmark_questions.json` is the single source of truth. 20 in-scope business
questions (one to four per paper) + 4 off-topic probes, each with:

```json
{
  "id": "q01",
  "query": "How can businesses reduce LLM costs?",
  "category": "cost",
  "expected_paper_id": "2608.20316",
  "expected_section_prefixes": ["Abstract", "1 Introduction"],
  "ground_truth_answer": "Short reference summary drafted from the paper.",
  "off_topic": false
}
```

To add a question: append an entry, give it the next `qNN` id, and re-run. The
root `eval_retrieval.py` (HyDE A/B comparison) loads the same file.

## Running

```bash
python eval/run_eval.py                 # full run (retrieval + generation + report)
python eval/run_eval.py --limit 3       # smoke test on 3 questions
python eval/run_eval.py --skip-generation   # retrieval only
python eval/evaluate_retrieval.py       # stages can also run individually
python eval/evaluate_generation.py      # requires GEMINI_API_KEY
python eval/make_report.py              # re-render report from existing results
```

Generation + judging makes ~3 Gemini calls per question with 4s pauses between
them (free-tier rate limits); a full 20-question run takes roughly 8-10 minutes.

## What each stage measures

| Stage | Metric | How |
|---|---|---|
| Retrieval | Paper Hit@1, Precision@k | Expected paper in top-k (deterministic) |
| Retrieval | Section Hit@1/@k, MRR, Recall@k | Hit's section_path starts with an expected prefix (deterministic) |
| Retrieval | Distance separation | Relevant vs. off-topic distances — is a cutoff feasible? |
| Generation | Faithfulness, citation validity | `reviewer_agent.review_answer` (Gemini judge) |
| Generation | Answer relevance 1-5, business quality 1-5 | Gemini judge with anchored rubrics |
| Generation | Citation coverage, invalid [n] markers | Mechanical parse of `[n]` vs. number of sources |

## Outputs

- `eval/results/retrieval_results.jsonl` — per-question retrieval results
- `eval/results/retrieval_metrics.json` — aggregate retrieval metrics
- `eval/results/generation_results.jsonl` — per-question answers, citations, judge scores
- `eval/results/generation_metrics.json` — aggregate generation metrics
- `eval/findings.md` — hand-curated findings, included in the report on every render
- `reports/evaluation_report.md` — human-readable report (rendered by `make_report.py`)

Curate the report's "Findings and improvement opportunities" section by editing
`eval/findings.md`; the rest of the report is generated.
