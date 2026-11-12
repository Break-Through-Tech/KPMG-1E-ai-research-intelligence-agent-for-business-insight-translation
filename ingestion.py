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


def _abstract_chunk(paper: dict) -> list[dict]:
    """Single abstract-only chunk, used when the PDF has no extractable text (e.g. a scan)."""
    abstract = (paper.get("abstract") or "").strip()
    if not abstract:
        return []
    return [{
        "chunk_id": f"{paper['paper_id']}_s00_c00",
        "paper_id": paper["paper_id"],
        "section_label": "Abstract",
        "section_number": "",
        "section_path": "Abstract",
        "chunk_index": 0,
        "n_chunks_in_section": 1,
        "chunk_text": abstract,
        "word_count": len(abstract.split()),
        "page_start": 1,
        "page_end": 1,
        "chunking_method": "abstract_only",
        "also_covers": "",
    }]


def chunk_paper(paper: dict) -> list[dict]:
    """Chunk the PDF at paper["pdf_path"] into chunk dicts in the chunk.py format.

    PDFs with no extractable text fall back to one chunk made from the arXiv abstract, so the
    paper is still searchable instead of being retried every day. Raises if nothing is usable.
    """
    import chunk as chunker  # lazy: pdfplumber is only needed when ingesting

    record, chunks = chunker.chunk_paper(paper["pdf_path"], title_override=paper.get("title"))
    if not chunks:
        logging.warning("no text chunks for %s (%s); using the abstract only",
                        paper["paper_id"], record.get("parse_status"))
        chunks = _abstract_chunk(paper)
    if not chunks:
        raise ValueError(f"no chunks and no abstract for {paper['paper_id']}")
    # chunk.py derives the id from the file name; a mismatch would index chunks under the wrong paper.
    wrong = {c["paper_id"] for c in chunks} - {paper["paper_id"]}
    if wrong:
        raise ValueError(f"chunks carry paper_id {sorted(wrong)} but the paper is {paper['paper_id']}")
    return chunks


def _published_ts(published: str) -> int:
    """'2026-08-20T...' -> 20260820 (0 if unknown); matches the filter in function_app._where."""
    digits = (published or "")[:10].replace("-", "")
    return int(digits) if digits.isdigit() else 0


def _vector_metadata(paper: dict, chunk: dict) -> dict:
    """Pinecone metadata for one chunk; the field names are what retrieve.search() reads back."""
    return {
        "paper_id": paper["paper_id"],
        "title": paper.get("title") or "",
        "authors": "; ".join(paper.get("authors") or []),
        "categories": " ".join(paper.get("categories") or []),
        "published": (paper.get("published") or "")[:10],
        "published_ts": _published_ts(paper.get("published") or ""),
        "pdf_url": paper.get("pdf_url") or "",
        "section_label": chunk["section_label"],
        "section_path": chunk.get("section_path", chunk["section_label"]),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "page_start": int(chunk.get("page_start", 0)),
        "page_end": int(chunk.get("page_end", 0)),
        "text": chunk["chunk_text"],
    }


def index_chunks(paper: dict, chunks: list[dict]) -> None:
    """Embed the chunks with Azure OpenAI and upsert them to Pinecone (ids = chunk_id, so reruns overwrite)."""
    if not chunks:
        return
    from azure_embed import embed_texts
    from pinecone import Pinecone

    vectors = embed_texts([c["chunk_text"] for c in chunks])
    index = Pinecone(api_key=config.PINECONE_API_KEY).Index(config.PINECONE_INDEX)
    records = [
        {"id": c["chunk_id"], "values": v, "metadata": _vector_metadata(paper, c)}
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(records), 100):
        index.upsert(vectors=records[i : i + 100], namespace=config.PINECONE_NAMESPACE)


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


def save_run_summary(summary: dict) -> None:
    """Keep a durable record of each run at runs/<UTC timestamp>.json so quiet days are visible."""
    try:
        name = f"runs/{datetime.now(timezone.utc):%Y-%m-%dT%H%M%S}.json"
        _container().upload_blob(name, json.dumps(summary), overwrite=True)
    except Exception:
        logging.exception("could not save the run summary")


# arXiv doesn't announce papers on weekends, so Sunday and Monday runs (06:00 UTC) normally find
# nothing new. A quiet run on any other day means the fetch step is probably broken.
QUIET_DAYS_OK = (6, 0)  # datetime.weekday(): Monday=0 ... Sunday=6


def check_run(summary: dict, today: datetime | None = None) -> None:
    """Log the run summary and raise if something needs attention.

    Raising marks the function execution as failed, so a single Azure Monitor alert on failed
    executions of daily_ingest covers both partial failures and unexpectedly empty days.
    """
    logging.info("ingestion summary: %s", json.dumps(summary))
    problems = []
    if summary["failed"]:
        problems.append(f"{len(summary['failed'])} paper(s) failed: {', '.join(summary['failed'])}")
    today = today or datetime.now(timezone.utc)
    if summary["fetched"] == 0 and today.weekday() not in QUIET_DAYS_OK:
        problems.append("no new papers were fetched")
    if problems:
        message = "daily ingestion needs attention: " + "; ".join(problems)
        logging.error(message)
        raise RuntimeError(message)


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
    save_run_summary(summary)
    return summary
