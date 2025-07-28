from pathlib import Path

# Paths
DATA_DIR = Path("data/documents")
OUTPUT_FILE = Path("output/challenge_output.json")
FAISS_INDEX = "retriever.index"

# Models
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "distilgpt2"  # lightweight, <1GB

# Chunking
CHUNK_SIZE = 300  # tokens
CHUNK_OVERLAP = 50

