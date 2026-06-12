VECTOR_DB_PATH = "vector_db"
MODEL_NAME     = "qwen2.5:7b"
TOP_K          = 8
DOCS_URL       = "https://docs.python.org/3/"
NO_INFO_PHRASE = "I don't have enough information in the documentation to answer this question."

HARD_THRESHOLD = 1.20   
EASY_THRESHOLD = 0.80
RERANK_CUTOFF  = 0.0    

CHUNKS_FOLDER   = "docs/chunks"
CHECKPOINT_FILE = "docs/checkpoint.json"
CHUNK_SIZE      = 1200
CHUNK_OVERLAP   = 200
BATCH_SIZE      = 50