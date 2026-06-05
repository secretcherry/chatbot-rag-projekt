import os
import json
import time
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKS_FOLDER   = "docs/chunks"
VECTOR_DB_PATH  = "vector_db"
CHECKPOINT_FILE = "docs/checkpoint.json"
CHUNK_SIZE    = 700
CHUNK_OVERLAP = 80   

BATCH_SIZE = 50

all_files = sorted(Path(CHUNKS_FOLDER).glob("*.txt"))

if not all_files:
    print(f"No files found in {CHUNKS_FOLDER}/")
    print("Run first: python load_docs.py")
    exit()

print("=" * 50)
print("Creating vector database")
print(f"Number of files: {len(all_files)}")
print(f"Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
print(f"Batch size: {BATCH_SIZE}")
print("=" * 50)

done_files = set()

if Path(CHECKPOINT_FILE).exists():
    with open(CHECKPOINT_FILE, "r") as f:
        data = json.load(f)
        done_files = set(data.get("done", []))

remaining = [f for f in all_files if f.name not in done_files]

if done_files:
    print(f"\nCheckpoint: {len(done_files)} files already processed")
    print(f"Remaining:  {len(remaining)} files\n")
else:
    print(f"\nStarting fresh - {len(remaining)} files\n")

if not remaining:
    print("Everything already processed! Database is ready.")
    exit()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)

print("Loading embedding model (may take a while the first time)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Model ready!\n")

db = None

if Path(VECTOR_DB_PATH).exists() and done_files:
    print("Loading existing FAISS database...")
    db = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Database loaded!\n")

total_chunks = 0
batch_num = 0

for start in range(0, len(remaining), BATCH_SIZE):
    batch = remaining[start : start + BATCH_SIZE]
    batch_num += 1

    end = min(start + BATCH_SIZE, len(remaining))
    print(f"Batch {batch_num} (files {start + 1}-{end} of {len(remaining)})")

    batch_docs = []

    for filepath in batch:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()

            metadata = {"section": "general", "title": "", "source": ""}
            separator_idx = 0

            for i, line in enumerate(lines):
                if line.startswith("SECTION:"):
                    metadata["section"] = line.split(":", 1)[1].strip()
                elif line.startswith("TITLE:"):
                    metadata["title"] = line.split(":", 1)[1].strip()
                elif line.startswith("SOURCE:"):
                    metadata["source"] = line.split(":", 1)[1].strip()
                elif line.startswith("-" * 20):
                    separator_idx = i + 1
                    break

            text = "".join(lines[separator_idx:]).strip()

            if len(text) < 100:
                done_files.add(filepath.name)
                continue

            chunks = splitter.create_documents(
                texts=[text],
                metadatas=[{
                    "source":   metadata["source"],
                    "title":    metadata["title"],
                    "section":  metadata["section"],
                    "filename": filepath.name,
                }]
            )

            batch_docs.extend(chunks)
            total_chunks += len(chunks)

        except Exception as e:
            print(f"  Error reading {filepath.name}: {e}")

    if not batch_docs:
        for f in batch:
            done_files.add(f.name)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"done": list(done_files)}, f, indent=2)
        continue

    try:
        print(f"  Embedding {len(batch_docs)} chunks...", end="", flush=True)
        t0 = time.time()

        if db is None:
            db = FAISS.from_documents(batch_docs, embeddings)
        else:
            new_db = FAISS.from_documents(batch_docs, embeddings)
            db.merge_from(new_db)

        elapsed = time.time() - t0
        print(f" done! ({elapsed:.1f}s)")

        db.save_local(VECTOR_DB_PATH)

        for f in batch:
            done_files.add(f.name)

        Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"done": list(done_files)}, f, indent=2)

        print(f"  Database saved. Total chunks so far: {total_chunks}\n")

    except Exception as e:
        print(f"\n  Error embedding batch {batch_num}: {e}")
        print("  Checkpoint saved - run the script again to continue.")
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({"done": list(done_files)}, f, indent=2)
        exit()

print("=" * 50)
print("DONE!")
print(f"Files processed: {len(done_files)}")
print(f"Total chunks:    {total_chunks}")
print(f"Database saved to: {VECTOR_DB_PATH}/")
print("=" * 50)