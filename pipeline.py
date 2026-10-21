import argparse
import subprocess
import sys

import config

CHUNKS_PATH = config.ROOT / "chunks" / "chunks.jsonl"


def run(step: str, script: str, *args: str) -> None:
    print(f"\n=== {step} ===", flush=True)
    result = subprocess.run([sys.executable, str(config.ROOT / script), *args])
    if result.returncode:
        sys.exit(f"Pipeline stopped: {step} failed (exit code {result.returncode}).")


def chunk_count() -> int:
    if not CHUNKS_PATH.exists():
        return 0
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Fetch arXiv papers, chunk new PDFs, and ingest them into ChromaDB.")
    parser.add_argument("--skip-fetch", action="store_true", help="Use the PDFs already in data/")
    parser.add_argument("--force-ingest", action="store_true", help="Ingest even if chunking added nothing new")
    fetch = parser.add_argument_group("fetch options (passed to fetch_arxiv.py)")
    fetch.add_argument("--category")
    fetch.add_argument("--max-results", type=int)
    fetch.add_argument("--days", type=int)
    fetch.add_argument("--since", metavar="YYYY-MM-DD")
    fetch.add_argument("--until", metavar="YYYY-MM-DD")
    fetch.add_argument("--keyword", action="append", default=[])
    args = parser.parse_args()

    if not args.skip_fetch:
        fetch_args = []
        for flag in ("category", "max_results", "days", "since", "until"):
            value = getattr(args, flag)
            if value is not None:
                fetch_args += [f"--{flag.replace('_', '-')}", str(value)]
        for keyword in args.keyword:
            fetch_args += ["--keyword", keyword]
        run("Fetch", "fetch_arxiv.py", *fetch_args)

    before = chunk_count()
    run("Chunk", "chunk.py", "--incremental")
    added = chunk_count() - before

    if added or args.force_ingest:
        run("Ingest", "ingest.py")
    else:
        print("\n=== Ingest ===\nNo new chunks; skipping (use --force-ingest to run anyway).")
    print(f"\nDone. {added} new chunk(s).")


if __name__ == "__main__":
    main()
