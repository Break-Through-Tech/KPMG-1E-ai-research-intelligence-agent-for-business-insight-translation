import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
# bge models are trained with this prefix on queries only; documents are embedded as-is.
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
QUERY_INSTRUCTION = os.getenv(
    "QUERY_INSTRUCTION", _BGE_QUERY_INSTRUCTION if "bge" in EMBEDDING_MODEL.lower() else ""
)
# Relative paths resolve against the repo root, so scripts and notebooks share one DB.
CHROMA_DB_PATH = str(ROOT / os.getenv("CHROMA_DB_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kpmg_papers")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "")

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
PDF_CONTAINER = os.getenv("PDF_CONTAINER", "pdfs")
