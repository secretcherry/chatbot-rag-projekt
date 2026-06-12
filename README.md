# Python RAG Chatbot

This is my final college project: a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Python 3 using the official Python documentation.

The chatbot is designed to answer **only documentation-related questions** and avoids hallucinations by generating responses solely from retrieved documentation content. If it cannot find relevant information, it explicitly states that instead of guessing.

---

## Features

* Answers questions about Python 3
* Searches the official Python documentation for relevant information
* Uses a local LLM via Ollama to generate responses based only on retrieved context
* Refuses to answer when no relevant documentation is found
* Provides clickable source links that automatically highlight the relevant text in the official documentation
* Supports multiple chat sessions through a Streamlit web interface

---

## Requirements

* Python 3.10 or newer
* Ollama installed and running
* Python 3 HTML documentation downloaded locally

---

## Installation

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the language model

```bash
ollama pull qwen2.5:7b
```

### 3. Download the Python documentation

Download the Python 3 HTML documentation from:

https://docs.python.org/3/download.html

Extract the files into a folder named:

```
python_docs/
```

located in the project root directory.

---

## Preparing the Knowledge Base

### 1. Parse the documentation

```bash
python load_docs.py
```

This script:

* Processes all HTML documentation files
* Removes navigation menus and unnecessary content
* Extracts useful text
* Saves the processed documents into:

```
docs/chunks/
```

### 2. Build the vector database

```bash
python create_vector_db.py
```

This script:

* Loads the parsed documentation chunks
* Splits documents while respecting Markdown structure
* Generates embeddings
* Builds and saves a FAISS vector index
* Periodically saves progress during processing

The generated database is stored in:

```
vector_db/
```

---

## Running the Chatbot

### Streamlit Web UI

```bash
streamlit run app.py
```

### Terminal Version

```bash
python rag_chatbot.py
```

---

## Running Tests

To verify the RAG engine's logic, including topic filtering and prompt generation:

```bash
pytest test_chatbot.py
```

---

## How It Works

1. The user's question is first checked by a keyword filter to reject obviously unrelated topics.
2. If previous conversation history exists, the query is rewritten into a standalone search query.
3. The rewritten query is used to search the FAISS vector database.
4. Retrieved documents are reranked using a CrossEncoder model.
5. The highest-ranked documents, together with chat history, are provided to the LLM.
6. The LLM generates an answer based only on the retrieved context.
7. If the required information is not available, the chatbot states that it does not know.

---

## Models Used

| Component  | Model                                  |
| ---------- | -------------------------------------- |
| LLM        | `qwen2.5:7b` (via Ollama)              |
| Embeddings | `BAAI/bge-base-en-v1.5`                |
| Reranker   | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

---

## Project Structure

```text
├── app.py                  # Streamlit web UI
├── rag_chatbot.py          # Terminal chatbot
├── rag_engine.py           # Core RAG pipeline
├── config.py               # Global settings and thresholds
├── test_chatbot.py         # Test suite
├── load_docs.py            # HTML parser
├── create_vector_db.py     # FAISS index builder
├── requirements.txt
├── python_docs/            # Python HTML documentation
├── docs/
│   └── chunks/             # Parsed documentation chunks
└── vector_db/              # Generated FAISS index
```

---

## Tech Stack

* Python
* Streamlit
* Ollama
* FAISS
* Sentence Transformers
* CrossEncoder reranking
* Retrieval-Augmented Generation (RAG)

---

## Purpose

This project was developed as a final college project to demonstrate the implementation of a local Retrieval-Augmented Generation system that combines document retrieval, semantic search, reranking, and local large language models to provide accurate, source-grounded answers from the official Python documentation.
