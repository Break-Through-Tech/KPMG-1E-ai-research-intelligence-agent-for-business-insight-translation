import json
import logging
import os
import time
from dataclasses import asdict
from datetime import date

import azure.functions as func

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

MAX_K = 20
MODES = ("answer", "insight")


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health(req: func.HttpRequest) -> func.HttpResponse:
    return func.HttpResponse("ok")


def _json(body: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(json.dumps(body), status_code=status, mimetype="application/json")


def _error(status: int, message: str) -> func.HttpResponse:
    return _json({"error": message}, status)


def _where(filters: dict) -> dict | None:
    """Turn request filters into a metadata filter (same syntax in Chroma and Pinecone)."""
    clauses = []
    if filters.get("paper_ids"):
        clauses.append({"paper_id": {"$in": list(filters["paper_ids"])}})
    if filters.get("published_after"):
        day = date.fromisoformat(filters["published_after"])
        clauses.append({"published_ts": {"$gte": int(day.strftime("%Y%m%d"))}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _insight(question: str, k: int) -> func.HttpResponse:
    """Business-insight mode: hand the question to the agent workflow in agents.py.

    agents.run_pipeline(question, k, mode="insight") must return a dataclass with question,
    status, answer, review, insight, sources, timings and trace (see InsightResult). It does its
    own retrieval, so filters and HyDE don't apply to this mode yet.
    """
    try:
        from agents import run_pipeline
    except ImportError:
        return _error(501, "Insight mode isn't available yet.")
    try:
        result = run_pipeline(question, k=k, mode="insight")
    except Exception:
        logging.exception("insight workflow failed")
        return _error(503, "The insight workflow failed. Try again later.")
    return _json(asdict(result))


@app.route(route="query", methods=["POST"])
def query(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
    except ValueError:
        return _error(400, "Request body must be JSON.")
    if not isinstance(body, dict):
        return _error(400, "Request body must be a JSON object.")

    question = str(body.get("question") or "").strip()
    if not question:
        return _error(400, "'question' is required.")
    k = body.get("k", 5)
    if not isinstance(k, int) or not 1 <= k <= MAX_K:
        return _error(400, f"'k' must be an integer between 1 and {MAX_K}.")
    use_hyde = bool(body.get("use_hyde", False))
    mode = body.get("mode", "answer")
    if mode not in MODES:
        return _error(400, f"'mode' must be one of {list(MODES)}.")
    filters = body.get("filters") or {}
    if not isinstance(filters, dict):
        return _error(400, "'filters' must be an object.")
    try:
        where = _where(filters)
    except (TypeError, ValueError):
        return _error(400, "'filters.published_after' must be a YYYY-MM-DD date and 'paper_ids' a list.")

    if mode == "insight":
        return _insight(question, k)

    # Imported per request so a broken model or vector store can't stop /api/health from loading.
    from generate import generate_answer
    from retrieve import retrieve
    from reviewer_agent import review_answer

    timings = {}
    start = time.perf_counter()
    try:
        hits = retrieve(question, k=k, use_hyde=use_hyde, where=where)
    except Exception:
        logging.exception("retrieval failed")
        return _error(503, "Search is unavailable right now. Try again later.")
    timings["retrieve"] = round(time.perf_counter() - start, 2)

    response = {"question": question, "answer": "", "review": None, "insight": None,
                "sources": hits, "timings": timings}
    if not hits:
        response["answer"] = "No relevant papers were found for this question."
        return _json(response)

    start = time.perf_counter()
    answer = generate_answer(question, hits)
    timings["generate"] = round(time.perf_counter() - start, 2)
    if not answer:
        return _error(503, "Answer generation failed (the Gemini quota may be used up). Try again later.")
    response["answer"] = answer

    start = time.perf_counter()
    try:
        response["review"] = review_answer(hits, answer)
    except Exception:
        logging.exception("review failed")
    timings["review"] = round(time.perf_counter() - start, 2)

    return _json(response)


@app.timer_trigger(schedule="0 0 6 * * *", arg_name="timer", run_on_startup=False, use_monitor=True)
def daily_ingest(timer: func.TimerRequest) -> None:
    """Pull new arXiv papers into Pinecone and Blob Storage every day at 06:00 UTC."""
    from ingestion import check_run, run_ingestion

    try:
        # ARCHIVE_ONLY=1 stores PDFs under staging/ and skips chunking/indexing (see ingestion.py).
        summary = run_ingestion(archive_only=os.getenv("ARCHIVE_ONLY") == "1")
    except NotImplementedError:
        logging.warning("Ingestion steps aren't implemented yet; skipping this run.")
        return
    check_run(summary)  # logs the summary; raises on failed papers or an unexpectedly empty day
