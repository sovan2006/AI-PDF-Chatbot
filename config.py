import os

# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIRECTORY = os.path.dirname(
    os.path.abspath(__file__)
)


# ============================================================
# STORAGE DIRECTORIES
# ============================================================

UPLOAD_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "uploads"
)

VECTORSTORE_DIRECTORY = os.path.join(
    BASE_DIRECTORY,
    "vectorstore"
)


# ============================================================
# DOCUMENT CHUNKING CONFIGURATION
# ============================================================

CHUNK_SIZE = 1000

CHUNK_OVERLAP = 200


# ============================================================
# EMBEDDING CONFIGURATION
# ============================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# VECTOR STORE CONFIGURATION
# ============================================================

COLLECTION_NAME = "pdf_documents"


# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 5

HYBRID_CANDIDATE_K = 20

RRF_K = 60


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(
    UPLOAD_DIRECTORY,
    exist_ok=True
)

os.makedirs(
    VECTORSTORE_DIRECTORY,
    exist_ok=True
)
# ============================================================
# RETRIEVAL CONFIGURATION
# ============================================================

TOP_K = 5

HYBRID_CANDIDATE_K = 20

DENSE_WEIGHT = 0.7

BM25_WEIGHT = 0.3

RRF_K = 60