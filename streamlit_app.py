"""Streamlit demo frontend for the AI Research Intelligence Agent (RAG over arXiv papers).

Reuses the repo's existing pipeline unchanged:
  retrieve.retrieve -> generate.generate_answer -> reviewer_agent.review_answer

Run locally:   streamlit run streamlit_app.py
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="AI Research Intelligence Agent",
    page_icon="🔬",
    layout="wide",
)


# --- API key resolution -----------------------------------------------------
# Prefer Streamlit Secrets (Community Cloud) over .env. Must happen BEFORE we
# import config, which reads GEMINI_API_KEY at import time.
def _from_secrets(name: str):
    try:
        if name in st.secrets:  # raises if no secrets.toml exists at all
            return str(st.secrets[name])
    except Exception:
        return None
    return None


for _key in ("GEMINI_API_KEY", "GEMINI_MODEL"):
    _val = _from_secrets(_key)
    if _val:
        os.environ[_key] = _val

# Project modules (import after env is set). We import whole modules so the
# demo can reset their session-level "disabled" flags without touching behavior
# for the CLI / Azure / eval scripts that never import this file.
import config  # noqa: E402
import store  # noqa: E402
import embed  # noqa: E402
import generate  # noqa: E402
import reviewer_agent  # noqa: E402
import retrieve as retrieve_mod  # noqa: E402

ROOT = config.ROOT
PY = sys.executable  # same interpreter/venv running Streamlit

EXAMPLE_QUESTIONS = [
    "How can businesses cut the cost of running large language models?",
    "How can AI learn repeatable workflows by watching how people use their computers?",
    "Can AI agents improve travel-demand forecasting and survey data collection?",
    "How can AI make medical report interpretation more reliable for patients?",
]


# --- Cached resources -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def warm_embedding_model():
    """Load the sentence-transformer once per process (the slow first-boot cost)."""
    return embed.get_model()


@st.cache_resource(show_spinner=False)
def get_collection():
    """One Chroma collection handle per process; wraps store.get_collection()."""
    return store.get_collection()


@st.cache_data(show_spinner=False)
def index_summary():
    """Chunk count and per-paper breakdown, read straight from the collection."""
    col = get_collection()
    count = col.count()
    papers: dict[str, dict] = {}
    if count:
        data = col.get(include=["metadatas"])
        for meta in data["metadatas"]:
            pid = meta.get("paper_id") or "?"
            entry = papers.setdefault(
                pid, {"paper_id": pid, "title": meta.get("title") or pid, "n_chunks": 0}
            )
            entry["n_chunks"] += 1
    return {"count": count, "papers": sorted(papers.values(), key=lambda p: p["title"])}


def refresh_index_caches():
    index_summary.clear()
    get_collection.clear()


# --- Gemini error helpers ---------------------------------------------------
def friendly_gemini_error(exc: Exception) -> str:
    text = str(exc)
    low = text.lower()
    if "resource_exhausted" in low or "429" in low or "rate" in low or "quota" in low:
        return (
            "The Gemini API is rate-limited right now (free-tier allows ~15 requests/min). "
            "Wait a few seconds and try again."
        )
    if "api key" in low or "api_key" in low or "permission" in low or "401" in low or "403" in low:
        return "The Gemini API key looks invalid or lacks permission. Check GEMINI_API_KEY."
    return f"Gemini call failed: {text[:300]}"


def safe_generate(question: str, hits: list[dict]):
    """Returns (answer, error_message)."""
    if not config.GEMINI_API_KEY:
        return "", "No GEMINI_API_KEY is set. Add it to .env (local) or Streamlit Secrets (cloud) to enable answers."
    try:
        answer = generate.generate_answer(question, hits)
    except Exception as exc:  # generate_answer normally swallows these, but be safe
        return "", friendly_gemini_error(exc)
    if not answer:
        return "", (
            "The model returned an empty answer — usually a transient API error or rate limit. "
            "Try again in a few seconds."
        )
    return answer, None


def safe_review(hits: list[dict], answer: str):
    """Returns (review_dict_or_None, error_message)."""
    try:
        return reviewer_agent.review_answer(hits, answer), None
    except Exception as exc:
        return None, friendly_gemini_error(exc)


# --- First-boot: build the index if it's empty ------------------------------
def ensure_index():
    warm_embedding_model()
    summary = index_summary()
    if summary["count"] > 0:
        return summary
    with st.spinner("First run: building the search index from chunks/ … this can take 1–2 minutes."):
        proc = subprocess.run([PY, str(ROOT / "ingest.py")], capture_output=True, text=True, cwd=str(ROOT))
    refresh_index_caches()
    summary = index_summary()
    if summary["count"] == 0:
        st.error("Could not build the index automatically.")
        st.code((proc.stderr or proc.stdout or "")[-2000:])
        st.stop()
    return summary


# --- Pipeline runner (Ask tab) ----------------------------------------------
def run_pipeline(question: str, k: int, use_hyde: bool, where: dict | None):
    # Clear any sticky per-session "disabled" flags so one earlier transient
    # error doesn't silently mute generation/HyDE for the rest of the demo.
    generate._gen_disabled_reason = ""
    retrieve_mod._hyde_disabled_reason = ""

    result: dict = {"question": question, "k": k, "use_hyde": use_hyde}
    timings: dict[str, float] = {}

    with st.status("Running the pipeline…", expanded=True) as status:
        # 1. Retrieve
        st.write("🔎 Retrieving sources…")
        t = time.perf_counter()
        try:
            hits = retrieve_mod.retrieve(question, k=k, use_hyde=use_hyde, where=where)
        except Exception as exc:
            status.update(label="Retrieval failed", state="error")
            st.error(f"Retrieval failed: {exc}")
            return
        timings["Retrieve"] = time.perf_counter() - t
        st.write(f"Found {len(hits)} source(s) in {timings['Retrieve']:.2f}s.")
        result["hits"] = hits

        if not hits:
            status.update(label="No matching sources", state="error")
            result.update(answer="", review=None, gen_error=None, review_error=None, timings=timings)
            st.session_state.last_result = result
            return

        # 2. Generate
        st.write("✍️ Generating the cited answer…")
        t = time.perf_counter()
        answer, gen_error = safe_generate(question, hits)
        timings["Generate"] = time.perf_counter() - t
        st.write(f"Answer stage done in {timings['Generate']:.2f}s.")
        result["answer"] = answer
        result["gen_error"] = gen_error

        # 3. Review (only if we have an answer to review)
        review, review_error = None, None
        if answer:
            st.write("🔬 Running the reviewer agent…")
            t = time.perf_counter()
            review, review_error = safe_review(hits, answer)
            timings["Review"] = time.perf_counter() - t
            st.write(f"Review stage done in {timings['Review']:.2f}s.")
        result["review"] = review
        result["review_error"] = review_error
        result["timings"] = timings

        status.update(label=f"Done in {sum(timings.values()):.2f}s", state="complete")

    st.session_state.last_result = result


# --- Renderers --------------------------------------------------------------
def _pages_label(hit: dict) -> str:
    start, end = hit.get("page_start"), hit.get("page_end")
    if not start:
        return ""
    return f"p.{start}" + (f"–{end}" if end and end != start else "")


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _format_published(published: str | None) -> str:
    """'2026-08-20' -> 'Aug 2026'. Falls back to the raw value if it doesn't parse."""
    if not published:
        return ""
    parts = str(published).split("-")
    try:
        return f"{_MONTHS[int(parts[1]) - 1]} {parts[0]}"
    except (IndexError, ValueError):
        return str(published)


def _format_authors(authors: str | None) -> str:
    """Semicolon-joined author string -> 'First Author et al.' when there's more than one."""
    if not authors:
        return ""
    names = [a.strip() for a in str(authors).split(";") if a.strip()]
    if not names:
        return ""
    return names[0] + (" et al." if len(names) > 1 else "")


def render_review(result: dict):
    error = result.get("review_error")
    review = result.get("review")
    if error:
        st.warning(f"Reviewer unavailable: {error}")
        return
    if not review:
        st.caption("No review — the reviewer runs only on a generated answer.")
        return

    # Expected shape from reviewer_agent: {faithful: bool, citations_valid: bool, issues: list}
    if (
        isinstance(review, dict)
        and isinstance(review.get("faithful"), bool)
        and isinstance(review.get("citations_valid"), bool)
    ):
        faithful = review["faithful"]
        cites_valid = review["citations_valid"]
        issues = review.get("issues") or []
        if faithful and cites_valid:
            st.success("✅ Faithful to the sources and all citations valid.")
        elif not faithful:
            st.error("❌ Reviewer flagged unsupported or overstated claims.")
        else:
            st.warning("⚠️ Reviewer flagged citation problems.")
        c1, c2 = st.columns(2)
        c1.metric("Faithful", "Yes" if faithful else "No")
        c2.metric("Citations valid", "Yes" if cites_valid else "No")
        if issues:
            st.markdown("**Issues raised:**")
            for item in issues:
                st.markdown(f"- {item}")
        else:
            st.caption("No issues reported.")
    else:
        # Unexpected shape — fall back to raw JSON so nothing is hidden.
        st.json(review)


def render_sources(hits: list[dict]):
    st.caption("Numbers match the [n] citations above. Distance is cosine distance — lower = more similar.")
    for i, hit in enumerate(hits, 1):
        dist = float(hit.get("score", hit.get("distance", 0.0)) or 0.0)
        color = "green" if dist < 0.4 else "orange" if dist < 0.5 else "red"
        title = hit.get("title") or "Untitled"
        with st.container(border=True):
            # Title + compact distance badge on one line.
            st.markdown(f"**[{i}] {title}**  :{color}-badge[distance {dist:.3f}]")
            # Authors · published · section · pages · PDF — all on one line.
            section = hit.get("section_path") or hit.get("section_label") or ""
            parts = [p for p in (
                _format_authors(hit.get("authors")),
                _format_published(hit.get("published")),
                section,
                _pages_label(hit),
            ) if p]
            line = " · ".join(parts) or "—"
            if hit.get("pdf_url"):
                line += f" · [📄 PDF]({hit['pdf_url']})"
            st.caption(line)
            with st.expander("Show excerpt text"):
                st.write(hit.get("text") or "")


def render_result():
    result = st.session_state.get("last_result")
    if not result:
        st.info("Ask a question above to get a cited answer, a reviewer check, and the source excerpts.")
        return

    timings = result.get("timings") or {}
    if timings:
        cols = st.columns(len(timings) + 1)
        for col, (name, secs) in zip(cols, timings.items()):
            col.metric(name, f"{secs:.2f}s")
        cols[-1].metric("Total", f"{sum(timings.values()):.2f}s")

    st.subheader("Answer")
    if result.get("answer"):
        st.markdown(result["answer"])
    else:
        st.warning(result.get("gen_error") or "No answer was generated.")

    st.subheader("Human review check")
    render_review(result)

    hits = result.get("hits") or []
    st.subheader(f"Sources ({len(hits)})")
    if hits:
        render_sources(hits)


# --- Fetch tab subprocess streaming -----------------------------------------
def run_subprocess_live(args: list[str]) -> int:
    placeholder = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen(
        args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=str(ROOT)
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        placeholder.code("\n".join(lines[-400:]) or "…")
    proc.wait()
    return proc.returncode


# --- Eval tab helpers -------------------------------------------------------
RESULTS_DIR = ROOT / "eval" / "results"


def load_json_file(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_jsonl_file(path: Path, limit: int | None = None):
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                    if limit and len(rows) >= limit:
                        break
    except Exception:
        return []
    return rows


# ============================================================================
# Layout
# ============================================================================
st.title("🔬 AI Research Intelligence Agent")
st.caption("Turn the latest AI research papers into cited, business-ready insights.")

summary = ensure_index()

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    k = st.slider("Sources to retrieve (k)", 1, 10, 5)
    use_hyde = st.toggle(
        "Use HyDE query expansion",
        value=False,
        help="Generates a hypothetical answer passage and averages it into the query embedding. Needs the Gemini key.",
    )

    paper_titles = [p["title"] for p in summary["papers"]]
    selected_titles = st.multiselect("Filter by paper (optional)", paper_titles)
    where_filter = {"title": {"$in": selected_titles}} if selected_titles else None

    st.divider()
    st.subheader("Example questions")
    st.session_state.setdefault("question", "")
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        if st.button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.question = q

    st.divider()
    st.caption(f"📚 Index: **{summary['count']}** chunks · **{len(summary['papers'])}** papers")
    if not config.GEMINI_API_KEY:
        st.warning("No GEMINI_API_KEY — retrieval works, but answers/review are disabled.")


tab_ask, tab_fetch, tab_eval = st.tabs(["Ask", "Fetch papers", "Evaluation"])

# --- Tab 1: Ask ---
with tab_ask:
    question = st.text_area(
        "Your question",
        key="question",
        height=100,
        placeholder="e.g. How can businesses cut the cost of running large language models?",
    )
    ask = st.button("Ask", type="primary")
    if ask and question.strip():
        run_pipeline(question.strip(), k, use_hyde, where_filter)
    elif ask:
        st.warning("Type a question first (or click an example in the sidebar).")
    st.divider()
    render_result()

# --- Tab 2: Fetch papers ---
with tab_fetch:
    st.markdown("Pull new papers from arXiv, chunk them, and ingest into the index. Mirrors `pipeline.py`.")
    c1, c2, c3 = st.columns(3)
    category = c1.text_input("Category", value="cs.AI")
    max_results = c2.number_input("Max results", min_value=1, max_value=500, value=25, step=1)
    days = c3.number_input("Days back", min_value=0, max_value=90, value=2, step=1)
    keyword = st.text_input("Keywords (comma separated, optional)", placeholder="agents, routing")
    cc1, cc2 = st.columns(2)
    skip_fetch = cc1.checkbox("Skip fetch (use PDFs already in data/)")
    force_ingest = cc2.checkbox("Force ingest (even if no new chunks)")

    st.info("Fetching hits the arXiv API with a ~3s courtesy delay per download, so a real fetch can take a while.")
    if st.button("Run pipeline", type="primary", key="run_pipeline_btn"):
        args = [PY, str(ROOT / "pipeline.py")]
        if skip_fetch:
            args.append("--skip-fetch")
        if force_ingest:
            args.append("--force-ingest")
        if not skip_fetch:
            args += ["--category", category, "--max-results", str(int(max_results)), "--days", str(int(days))]
            for kw in (w.strip() for w in keyword.split(",")):
                if kw:
                    args += ["--keyword", kw]
        with st.status("Running pipeline.py…", expanded=True) as status:
            code = run_subprocess_live(args)
            if code == 0:
                status.update(label="Pipeline finished", state="complete")
            else:
                status.update(label=f"Pipeline exited with code {code}", state="error")
        refresh_index_caches()
        new_summary = index_summary()
        st.success(f"Index refreshed: {new_summary['count']} chunks · {len(new_summary['papers'])} papers now searchable.")

# --- Tab 3: Evaluation ---
with tab_eval:
    ret_metrics = load_json_file(RESULTS_DIR / "retrieval_metrics.json")
    gen_metrics = load_json_file(RESULTS_DIR / "generation_metrics.json")

    if not ret_metrics and not gen_metrics:
        st.info("No evaluation results found yet.")
        st.markdown(
            "Generate them (see `eval/README.md`):\n\n"
            "```bash\n"
            "python eval/run_eval.py              # full run (retrieval + generation + report)\n"
            "python eval/run_eval.py --limit 3    # quick smoke test\n"
            "python eval/run_eval.py --skip-generation   # retrieval only (no Gemini)\n"
            "```"
        )
    else:
        if ret_metrics:
            st.subheader("Retrieval")
            cols = st.columns(4)
            cols[0].metric("Questions", ret_metrics.get("n_questions", "—"))
            cols[1].metric("Paper Hit@1", f"{ret_metrics.get('paper Hit@1', 0):.2f}")
            cols[2].metric("Section MRR", f"{ret_metrics.get('section MRR', 0):.2f}")
            cols[3].metric("Sec Recall@k", f"{ret_metrics.get('section Recall@k', 0):.2f}")
            st.dataframe(
                [{"metric": m, "value": v} for m, v in ret_metrics.items()],
                use_container_width=True,
                hide_index=True,
            )

        if gen_metrics:
            st.subheader("Generation")
            cols = st.columns(4)
            cols[0].metric("Faithful %", gen_metrics.get("faithful_pct", "—"))
            cols[1].metric("Citations valid %", gen_metrics.get("citations_valid_pct", "—"))
            cols[2].metric("Answer relevance /5", gen_metrics.get("mean_answer_relevance_1_5", "—"))
            cols[3].metric("Business quality /5", gen_metrics.get("mean_business_quality_1_5", "—"))
            st.dataframe(
                [{"metric": m, "value": v} for m, v in gen_metrics.items()],
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Sample results")
        ret_rows = load_jsonl_file(RESULTS_DIR / "retrieval_results.jsonl", limit=5)
        gen_rows = load_jsonl_file(RESULTS_DIR / "generation_results.jsonl", limit=5)
        if ret_rows:
            st.markdown("**Retrieval results (first 5)**")
            st.dataframe(ret_rows, use_container_width=True, hide_index=True)
        if gen_rows:
            st.markdown("**Generation results (first 5)**")
            st.dataframe(gen_rows, use_container_width=True, hide_index=True)
