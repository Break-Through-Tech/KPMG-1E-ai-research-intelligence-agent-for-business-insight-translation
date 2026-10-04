import os

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kpmg_papers")
