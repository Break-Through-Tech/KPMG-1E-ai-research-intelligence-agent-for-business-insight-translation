"""Daily ingestion job, run by the `daily_ingest` timer in function_app.py.

Each run downloads new arXiv papers into a temporary folder, then handles one paper at a time:
chunk it, embed and upsert the chunks to Pinecone, and finally archive the PDF to Blob Storage.
The function app's disk is small and wiped between runs, so each PDF is deleted once handled.
Archiving runs last so a PDF in Blob Storage means the paper was fully ingested; a paper that
fails part-way is retried on the next run.
"""
import logging
import tempfile
import time
from pathlib import Path


def fetch_new_papers(work_dir: Path, days: int, max_results: int) -> list[dict]:
    """Download papers submitted in the last `days` days that aren't ingested yet into `work_dir`.

    "Already ingested" = papers/{paper_id}.pdf exists in Blob Storage (the local data/ folder
    doesn't exist in Azure). Returns one fetch_arxiv.py metadata record per paper, plus
    "pdf_path" pointing at the downloaded file. Owner: arXiv fetching.
    """
    raise NotImplementedError


def chunk_paper(paper: dict) -> list[dict]:
    """Chunk the PDF at paper["pdf_path"] into chunk dicts in the chunk.py format. Owner: chunking."""
    raise NotImplementedError


def index_chunks(paper: dict, chunks: list[dict]) -> None:
    """Embed the chunks with Azure OpenAI and upsert them to Pinecone. Owner: Task 1."""
    raise NotImplementedError


def archive_paper(paper: dict) -> None:
    """Upload the PDF to papers/{paper_id}.pdf and the record to meta/{paper_id}.json. Owner: Task 2."""
    raise NotImplementedError


def run_ingestion(days: int = 2, max_results: int = 50) -> dict:
    start = time.perf_counter()
    summary = {"fetched": 0, "ingested": 0, "chunks": 0, "failed": []}
    with tempfile.TemporaryDirectory(prefix="ingest-") as tmp:
        papers = fetch_new_papers(Path(tmp), days, max_results)
        summary["fetched"] = len(papers)
        for paper in papers:
            try:
                chunks = chunk_paper(paper)
                index_chunks(paper, chunks)
                archive_paper(paper)
            except NotImplementedError:
                raise
            except Exception:
                logging.exception("ingestion failed for %s", paper["paper_id"])
                summary["failed"].append(paper["paper_id"])
                continue
            finally:
                Path(paper["pdf_path"]).unlink(missing_ok=True)
            summary["ingested"] += 1
            summary["chunks"] += len(chunks)
    summary["seconds"] = round(time.perf_counter() - start, 1)
    return summary
