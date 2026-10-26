"""Run the full evaluation: retrieval stage, generation stage, then render the report.

    python eval/run_eval.py                # full run (~10-15 min with Gemini judging)
    python eval/run_eval.py --limit 3     # smoke test on 3 questions
"""
import argparse
import runpy
import subprocess
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR.parent))

import config  # noqa: E402
import store  # noqa: E402


def run_stage(script: str, args: list[str]) -> None:
    print(f"\n=== {script} ===", flush=True)
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / script), *args],
        cwd=EVAL_DIR.parent,
    )
    if result.returncode:
        sys.exit(f"Evaluation stopped: {script} failed (exit code {result.returncode}).")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, help="Only evaluate the first N in-scope questions")
    ap.add_argument("--skip-generation", action="store_true", help="Retrieval stage only")
    args = ap.parse_args()

    collection = store.get_collection()
    if not collection.count():
        sys.exit(
            "ChromaDB collection is empty. Run `python pipeline.py` (or python ingest.py) first."
        )
    print(f"ChromaDB collection: {collection.count()} chunks")

    stage_args = ["--k", str(args.k)]
    if args.limit:
        stage_args += ["--limit", str(args.limit)]

    run_stage("evaluate_retrieval.py", stage_args)

    if args.skip_generation:
        print("\n(--skip-generation: stopping after the retrieval stage)")
    elif not config.GEMINI_API_KEY:
        print(
            "\nGEMINI_API_KEY not set: skipping the generation stage "
            "(retrieval-only run; set it in .env to judge answers)."
        )
    else:
        run_stage("evaluate_generation.py", stage_args)

    print("\n=== make_report.py ===")
    run_stage("make_report.py", [])
    print("\nDone. Report: reports/evaluation_report.md")


if __name__ == "__main__":
    main()
