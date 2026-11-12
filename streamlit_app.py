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

import requests

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


for _key in ("GEMINI_API_KEY", "GEMINI_MODEL", "BACKEND_API_URL", "BACKEND_API_KEY"):
    _val = _from_secrets(_key)
    if _val:
        os.environ[_key] = _val

# Project modules (import after env is set). We import whole modules so the
# demo can reset their session-level "disabled" flags without touching behavior
# for the CLI / Azure / eval scripts that never import this file.

import config  # noqa: E402
if not os.environ.get("BACKEND_API_URL"):
    # Local Chroma + bge helpers (chromadb / sentence-transformers). With a backend configured
    # the app only needs the light cloud dependencies in requirements.txt, so skip them.
    import store  # noqa: E402
    import embed  # noqa: E402
import generate  # noqa: E402
import front_office_agent  # noqa: E402
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
            meta = meta or {}
            pid = meta.get("paper_id") or "?"
            entry = papers.setdefault(
                pid, {
                    "paper_id": pid,
                    "title": meta.get("title") or pid,