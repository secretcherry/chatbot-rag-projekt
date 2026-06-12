import urllib.parse
import ollama
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

import config

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
try:
    db = FAISS.load_local(
        config.VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
except Exception:
    db = None 

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def is_clearly_offtopic(q: str) -> bool:
    off_topic = [
        "what time", "what is the time", "current time",
        "what is a cat", "what is a dog", "what is a bird", "what is an animal",
        "weather", "recipe", "cook", "sport", "movie", "music", "football"
    ]
    return any(kw in q.lower() for kw in off_topic)

def check_relevance_with_llm(question: str) -> bool:
    prompt = (
        "Is the following question related to Python programming, Python syntax, "
        "Python libraries, or Python documentation?\n"
        "Answer with exactly one word: YES or NO.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    r = ollama.chat(model=config.MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return r["message"]["content"].strip().upper().startswith("YES")

def get_standalone_question(history, current_q: str) -> str:
    if not history:
        return current_q
    hist_text = ""
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        hist_text += f"{role}: {msg['content']}\n"
    prompt = (
        "Rewrite the user's question as a short standalone search query (max 10 words).\n"
        "Return ONLY the rewritten question, nothing else.\n\n"
        f"Chat History:\n{hist_text}\n"
        f"User's Question: {current_q}\n\nStandalone question:"
    )
    r = ollama.chat(model=config.MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    rewritten = r["message"]["content"].strip().splitlines()[0].strip().strip("'\"")
    return rewritten if len(rewritten.split()) <= 15 else current_q

def retrieve_documents(search_query: str, original_question: str):
    if db is None:
        return []

    if is_clearly_offtopic(original_question):
        return []

    faiss_results = db.similarity_search_with_score(search_query, k=config.TOP_K)
    if not faiss_results:
        return []

    best_faiss_score = faiss_results[0][1]

    if best_faiss_score > config.HARD_THRESHOLD:
        return []
    elif best_faiss_score > config.EASY_THRESHOLD:
        if not check_relevance_with_llm(original_question):
            return []

    docs_only = [doc for doc, _ in faiss_results]
    pairs     = [(search_query, doc.page_content) for doc in docs_only]
    rscores   = reranker.predict(pairs)
    ranked    = sorted(zip(docs_only, rscores), key=lambda x: x[1], reverse=True)
    
    return [doc for doc, score in ranked if score > config.RERANK_CUTOFF][:3]

def get_system_prompt() -> str:
    return (
        "You are a helpful Python documentation assistant.\n\n"
        "RULES:\n"
        "1. You must base your answer ONLY on the provided Documentation Context.\n"
        f"2. If the context does not contain the answer, reply exactly: {config.NO_INFO_PHRASE}\n"
        "3. Keep answers clear, concise, and provide code examples if they exist in the context.\n"
        "4. Format any code examples strictly with Markdown using ```python\n"
    )

def format_sources(valid_docs) -> list:
    sources = []
    seen = []
    for doc in valid_docs:
        src = doc.metadata.get("source", "")
        if src in seen:
            continue
        seen.append(src)
        
        clean_text = " ".join(doc.page_content.split())
        encoded_snippet = urllib.parse.quote(clean_text[:50])
        highlight_url = f"{config.DOCS_URL}{src.replace(chr(92), '/')}#:~:text={encoded_snippet}"
        
        sources.append({
            "title":   doc.metadata.get("title", "No title"),
            "url":     highlight_url,
            "preview": clean_text[:200],
        })
    return sources