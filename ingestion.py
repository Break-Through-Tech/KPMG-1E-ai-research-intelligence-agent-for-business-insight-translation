"""Daily ingestion job, run by the `daily_ingest` timer in function_app.py.

Each run downloads new arXiv papers into a temporary folder, then handles one paper at a time:
chunk it, embed and upsert the chunks to Pinecone, and finally archive the PDF to Blob Storage.
The function app's disk is small and wiped between runs, so each PDF is deleted once handled.
Archiving runs last so a PDF in Blob Storage means the paper was fully ingested; a paper that
fails part-way is retried on the next run.

Archive-only mode (ARCHIVE_ONLY=1 on the function app) skips chunking and indexing and writes
to staging/ instead of papers/, so PDFs can pile up in Blob Storage before those steps exist
without being mistaken for fully ingested papers.
"""
import json
import logging
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import arxiv
from azure.storage.blob import BlobServiceClient

import config
import fetch_arxiv


def _container():
    service = BlobServiceClient.from_connection_string(config.STORAGE_CONNECTION_STRING)
    return service.get_container_client(config.PDF_CONTAINER)


def _blob_names(paper_id: str, archive_only: bool) -> tuple[str, str]:
    """(pdf blob, metadata blob) for a paper; staging/ holds papers that were only archived."""
    if archive_only:
        return f"staging/{paper_id}.pdf", f"staging/meta/{paper_id}.json"
    return f"papers/{paper_id}.pdf", f"meta/{paper_id}.json"


def fetch_new_papers(work_dir: Path, days: int, max_results: int, archive_only: bool = False) -> list[dict]:
    """Download papers submitted in the last `days` days that aren't ingested yet into `work_dir`.

    "Already ingested" = papers/{paper_id}.pdf exists in Blob Storage (the local data/ folder
    doesn't exist in Azure). In archive-only mode staging/{paper_id}.pdf also counts, so staged
    papers aren't downloaded again every day. Returns one fetch_arxiv.py metadata record per
    paper, plus "pdf_path" pointing at the downloaded file.
    """
    container = _container()
    end = datetime.now(timezone.utc)
    query = fetch_arxiv.build_query("cs.AI", end - timedelta(days=days), end, [])
    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in client.results(search):
        short_id = result.get_short_id()
        paper_id, version = short_id.rsplit("v", 1)
        done = [f"papers/{paper_id}.pdf"] + ([f"staging/{paper_id}.pdf"] if archive_only else [])
        if any(container.get_blob_client(name).exists() for name in done):
            continue
        path = work_dir / fetch_arxiv.safe_filename(result)
        try:
            fetch_arxiv.download_pdf(result.pdf_url, path)
        except Exception:
            # Drop any partial file; the paper has no blob yet so the next run retries it.
            path.unlink(missing_ok=True)
            logging.exception("download failed for %s", short_id)
            continue
        papers.append({
            "paper_id": paper_id,
            "version": version,
            "title": " ".join(result.title.split()),
            "abstract": " ".join(result.summary.split()),
            "authors": [a.name for a in result.authors],
            "categories": result.categories,
            "published": result.published.isoformat(),
            "pdf_url": result.pdf_url,
            "pdf_path": str(path),
        })
    return papers


def chunk_paper(paper: dict) -> list[dict]:
    """Chunk the PDF at paper["pdf_path"] into chunk dicts in the chunk.py format. Owner: chunking."""
    raise NotImplementedError


def index_chunks(paper: dict, chunks: list[dict]) -> None:
    """Embed the chunks with Azure OpenAI and upsert them to Pinecone. Owner: Task 1."""
    raise NotImplementedError


def archive_paper(paper: dict, archive_only: bool = False) -> None:
    """Upload the PDF to papers/{paper_id}.pdf and the record to meta/{paper_id}.json.

    In archive-only mode both go under staging/ instead. Owner: Task 2.
    """
    container = _container()
    pdf_blob, meta_blob = _blob_names(paper["paper_id"], archive_only)
    with open(paper["pdf_path"], "rb") as f:
        container.upload_blob(pdf_blob, f, overwrite=True)
    record = {k: v for k, v in paper.items() if k != "pdf_path"}
    # Metadata goes last so a meta blob implies the PDF upload finished.
    container.upload_blob(meta_blob, json.dumps(record, ensure_ascii=False), overwrite=True)


def run_ingestion(days: int = 2, max_results: int = 50, archive_only: bool = False) -> dict:
    start = time.perf_counter()
    summary = {"fetched": 0, "ingested": 0, "chunks": 0, "failed": [], "archive_only": archive_only}
    with tempfile.TemporaryDirectory(prefix="ingest-") as tmp:
        papers = fetch_new_papers(Path(tmp), days, max_results, archive_only)
        summary["fetched"] = len(papers)
        for paper in papers:
            chunks = []
            try:
                if not archive_only:
                    chunks = chunk_paper(paper)
                    index_chunks(paper, chunks)
                archive_paper(paper, archive_only)
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
