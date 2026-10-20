import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone

import arxiv

import config

DATA_DIR = config.ROOT / "data"
METADATA_PATH = DATA_DIR / "arxiv_metadata.jsonl"


def safe_filename(paper: arxiv.Result) -> str:
    """Match the existing '<id>v<n>_<title>.pdf' convention the chunking notebook reads."""
    title = re.sub(r"[^\w\s'-]", "", paper.title)
    title = re.sub(r"\s+", " ", title).strip()[:80]
    return f"{paper.get_short_id()}_{title}.pdf"


def known_paper_ids() -> set[str]:
    """Versionless IDs already fetched, so re-runs and new versions aren't re-downloaded."""
    ids = {p.name.split("_", 1)[0].rsplit("v", 1)[0] for p in DATA_DIR.glob("*.pdf")}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, encoding="utf-8") as f:
            ids.update(json.loads(line)["paper_id"] for line in f if line.strip())
    return ids


def parse_date(value: str) -> datetime:
    try:
        return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{value}' is not a YYYY-MM-DD date")


def build_query(category: str, start: datetime | None, end: datetime | None, keywords: list[str]) -> str:
    query = f"cat:{category}"
    if keywords:
        query += " AND (" + " OR ".join(f'abs:"{k}"' for k in keywords) + ")"
    if start:
        end = end or datetime.now(timezone.utc)
        query += f" AND submittedDate:[{start:%Y%m%d%H%M} TO {end:%Y%m%d%H%M}]"
    return query


def fetch(query: str, max_results: int | None, download: bool) -> list[dict]:
    # arXiv asks for ~1 request per 3 seconds; the client enforces this via delay_seconds.
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    DATA_DIR.mkdir(exist_ok=True)
    seen = known_paper_ids()
    new = []
    scanned = 0
    for paper in client.results(search):
        scanned += 1
        paper_id = paper.get_short_id().rsplit("v", 1)[0]
        if paper_id in seen:
            continue
        if download:
            filename = safe_filename(paper)
            try:
                paper.download_pdf(dirpath=str(DATA_DIR), filename=filename)
            except Exception as e:
                # Drop any partial file so the next run retries instead of treating it as fetched.
                (DATA_DIR / filename).unlink(missing_ok=True)
                print(f"failed {paper.get_short_id()}: {e}", file=sys.stderr)
                continue
        seen.add(paper_id)
        record = {
            "paper_id": paper_id,
            "version": paper.get_short_id().rsplit("v", 1)[1],
            "title": " ".join(paper.title.split()),
            "abstract": " ".join(paper.summary.split()),
            "authors": [a.name for a in paper.authors],
            "categories": paper.categories,
            "published": paper.published.isoformat(),
            "pdf_url": paper.pdf_url,
        }
        new.append(record)
        if download:
            # Written per paper so an interrupted backfill keeps metadata for every PDF on disk.
            with open(METADATA_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{'downloaded' if download else 'found'} {paper.get_short_id()}: {paper.title[:70]}")
    if max_results and scanned >= max_results:
        print(
            f"Warning: reached --max-results ({max_results}); results are newest-first, so older papers "
            "in the range may have been skipped. Raise --max-results (0 = no limit) or narrow the date range.",
            file=sys.stderr,
        )
    return new


def main():
    parser = argparse.ArgumentParser(description="Pull recent arXiv papers into data/.")
    parser.add_argument("--category", default="cs.AI")
    parser.add_argument("--max-results", type=int, default=200, help="Cap on papers scanned (0 = no limit)")
    window = parser.add_mutually_exclusive_group()
    window.add_argument("--days", type=int, default=2, help="Only papers submitted in the last N days (0 = no limit)")
    window.add_argument("--since", type=parse_date, metavar="YYYY-MM-DD", help="Backfill from this date (UTC) onward")
    parser.add_argument("--until", type=parse_date, metavar="YYYY-MM-DD",
                        help="End of the --since range, inclusive (default: now)")
    parser.add_argument("--keyword", action="append", default=[], help="Abstract keyword filter; repeatable")
    parser.add_argument("--dry-run", action="store_true", help="List matches without downloading PDFs")
    args = parser.parse_args()

    if args.until and not args.since:
        parser.error("--until requires --since")
    if args.since:
        start = args.since
        end = args.until + timedelta(days=1) if args.until else None  # --until is inclusive
        if end and end <= start:
            parser.error("--until must not be before --since")
    elif args.days:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=args.days)
    else:
        start = end = None

    query = build_query(args.category, start, end, args.keyword)
    print(f"Query: {query}")
    try:
        new = fetch(query, args.max_results or None, download=not args.dry_run)
    except arxiv.ArxivError as e:
        sys.exit(f"Error: arXiv request failed: {e}")
    print(f"{len(new)} new paper(s).")


if __name__ == "__main__":
    main()
