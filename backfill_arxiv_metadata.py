"""Rebuild data/arxiv_metadata.jsonl for the papers already chunked in chunks/papers.json.

The chunker (chunk.py) merges authors/published/pdf_url/categories/abstract from
data/arxiv_metadata.jsonl into each paper record. When that file is missing the
merge is a no-op and those fields come out blank. This looks the papers up by id
with the arxiv library and writes the same record format fetch_arxiv.py produces.

PDFs are NOT downloaded again.

    python backfill_arxiv_metadata.py
"""

import json
import sys
from pathlib import Path

import arxiv

import config

PAPERS_PATH = config.ROOT / "chunks" / "papers.json"
OUT_PATH = config.ROOT / "data" / "arxiv_metadata.jsonl"


def main() -> None:
    try:
        papers = json.loads(PAPERS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        sys.exit(f"Error: could not read {PAPERS_PATH}: {e}")

    paper_ids = [p["paper_id"] for p in papers if p.get("paper_id")]
    if not paper_ids:
        sys.exit("Error: no paper_ids in papers.json.")

    print(f"Looking up {len(paper_ids)} paper(s): {', '.join(paper_ids)}")
    # Same courtesy settings as fetch_arxiv.py (~1 request / 3s).
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(id_list=paper_ids)

    records = []
    found = set()
    try:
        for paper in client.results(search):
            short = paper.get_short_id()           # e.g. "2608.20316v1"
            pid, _, ver = short.rpartition("v")     # -> ("2608.20316", "v", "1")
            pid = pid or short
            records.append({
                "paper_id": pid,
                "version": ver,
                "title": " ".join(paper.title.split()),
                "abstract": " ".join(paper.summary.split()),
                "authors": [a.name for a in paper.authors],
                "categories": paper.categories,
                "published": paper.published.isoformat(),
                "pdf_url": paper.pdf_url,
            })
            found.add(pid)
            print(f"  found {short}: {paper.title[:70]}")
    except arxiv.ArxivError as e:
        sys.exit(f"Error: arXiv request failed: {e}")

    missing = [p for p in paper_ids if p not in found]
    if missing:
        print(f"WARNING: not found on arXiv: {', '.join(missing)}", file=sys.stderr)

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} record(s) to {OUT_PATH}")
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
