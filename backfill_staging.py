"""One-off backfill: ingest PDFs that archive-only runs left in staging/ and move them to papers/.

Run once the chunking and Pinecone steps in ingestion.py are implemented and ARCHIVE_ONLY is off:

    python backfill_staging.py --dry-run      # list what is staged
    python backfill_staging.py --limit 25     # process up to 25, then re-run for the rest

Each paper is chunked, indexed, archived to papers/{id}.pdf + meta/{id}.json, and only then removed
from staging/, so a failure part-way leaves it in staging/ for the next run.
"""
import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import ingestion

PREFIX = "staging/"
META_PREFIX = "staging/meta/"


def staged_ids(container) -> list[str]:
    """Paper IDs with a staged PDF (metadata is uploaded last, so it implies a complete upload)."""
    ids = []
    for blob in container.list_blobs(name_starts_with=PREFIX):
        name = blob.name
        if name.startswith(META_PREFIX) or not name.endswith(".pdf"):
            continue
        paper_id = name[len(PREFIX):-len(".pdf")]
        if container.get_blob_client(f"{META_PREFIX}{paper_id}.json").exists():
            ids.append(paper_id)
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs staged by archive-only runs.")
    parser.add_argument("--limit", type=int, default=0, help="Max papers to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="List staged papers without processing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    container = ingestion._container()
    ids = staged_ids(container)
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} staged paper(s).")
    if args.dry_run:
        for paper_id in ids:
            print(f"  {paper_id}")
        return

    failed = []
    with tempfile.TemporaryDirectory(prefix="backfill-") as tmp:
        for paper_id in ids:
            pdf_path = Path(tmp) / f"{paper_id.replace('/', '_')}.pdf"
            try:
                pdf_path.write_bytes(container.download_blob(f"{PREFIX}{paper_id}.pdf").readall())
                paper = json.loads(container.download_blob(f"{META_PREFIX}{paper_id}.json").readall())
                paper["pdf_path"] = str(pdf_path)

                chunks = ingestion.chunk_paper(paper)
                ingestion.index_chunks(paper, chunks)
                ingestion.archive_paper(paper)  # papers/ + meta/
                container.delete_blob(f"{PREFIX}{paper_id}.pdf")
                container.delete_blob(f"{META_PREFIX}{paper_id}.json")
                print(f"ingested {paper_id}: {len(chunks)} chunk(s)")
            except NotImplementedError:
                sys.exit("Chunking/indexing isn't implemented yet; nothing was moved out of staging/.")
            except Exception:
                logging.exception("backfill failed for %s", paper_id)
                failed.append(paper_id)
            finally:
                pdf_path.unlink(missing_ok=True)

    if failed:
        sys.exit(f"Error: {len(failed)} paper(s) failed and remain in staging/: {', '.join(failed)}")


if __name__ == "__main__":
    main()
