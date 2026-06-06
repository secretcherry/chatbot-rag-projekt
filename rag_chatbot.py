import sys
import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

VECTOR_DB_PATH = "vector_db"
MODEL_NAME     = "qwen2.5:7b"
TOP_K          = 8
DOCS_URL       = "https://docs.python.org/3/"
NO_INFO_PHRASE = "I don't have enough information in the documentation to answer this question."

HARD_THRESHOLD = 1.05   
EASY_THRESHOLD = 0.65   
RERANK_CUTOFF  = 0.0    

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Python RAG Chatbot")
print("Type 'exit' to quit\n")

chat_history = []


def get_standalone_question(history, current_q):
    if not history:
        return current_q

    history_text = ""
    for msg in history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Rewrite the user's question as a short standalone search query (max 10 words).
Return ONLY the rewritten question, nothing else. No explanation, no punctuation at the end.

Chat History:
{history_text}

User's Question: {current_q}

Standalone question:"""

    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    rewritten = response["message"]["content"].strip()
    rewritten = rewritten.splitlines()[0].strip().strip("'\"")

    if len(rewritten.split()) > 15:
        return current_q
    return rewritten


def is_clearly_offtopic(question: str) -> bool:
    off_topic = [
        "what time", "what is the time", "current time",
        "what is a cat", "what is a dog", "what is a bird", "what is an animal",
        "weather", "recipe", "cook", "sport", "movie", "music", "football",
        "who is", "where is", "when was",
    ]
    q = question.lower()
    return any(kw in q for kw in off_topic)


def check_relevance_with_llm(question: str) -> bool:
    prompt = f"""Is the following question related to Python programming, Python syntax, Python libraries, or Python documentation?
Answer with exactly one word: YES or NO.

Question: {question}

Answer:"""
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip().upper().startswith("YES")


def no_info():
    print(f"\nBot:\n{NO_INFO_PHRASE}\n")
    return NO_INFO_PHRASE


while True:
    question = input("You: ").strip()

    if question.lower() == "exit":
        break
    if not question:
        continue

    if is_clearly_offtopic(question):
        ans = no_info()
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": ans})
        continue

    search_query = question
    if chat_history:
        print("Thinking... understanding context")
        search_query = get_standalone_question(chat_history, question)

    print(f'Searching database for: "{search_query}"')

    faiss_results = db.similarity_search_with_score(search_query, k=TOP_K)

    if not faiss_results:
        ans = no_info()
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": ans})
        continue

    best_faiss_score = faiss_results[0][1]

    if best_faiss_score > HARD_THRESHOLD:
        ans = no_info()
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": ans})
        continue
    elif best_faiss_score > EASY_THRESHOLD:
        if not check_relevance_with_llm(question):
            ans = no_info()
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": ans})
            continue

    docs_only = [doc for doc, _ in faiss_results]
    pairs = [(search_query, doc.page_content) for doc in docs_only]
    rerank_scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(docs_only, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    valid_docs = [doc for doc, score in ranked_docs if score > RERANK_CUTOFF][:3]

    if not valid_docs:
        ans = no_info()
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": ans})
        continue

    context_parts = []
    for i, doc in enumerate(valid_docs, start=1):
        title = doc.metadata.get("title", "Unknown")
        context_parts.append(f"[Source {i}: {title}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    system_prompt = f"""You are a Python documentation assistant.

STRICT RULES:
- Use ONLY information from the Documentation Context.
- Answer ONLY the user's question.
- Do not invent information or use outside knowledge.
- If the answer is not in the context, reply exactly: {NO_INFO_PHRASE}
- Keep answers short and factual."""

    user_prompt = f"""Documentation Context:
{context}

Question: {search_query}

Answer (use ONLY the context above):"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]

    print("\nBot:")
    try:
        response_stream = ollama.chat(model=MODEL_NAME, messages=messages, stream=True)
        full_response = ""
        for chunk in response_stream:
            content = chunk["message"]["content"]
            full_response += content
            sys.stdout.write(content)
            sys.stdout.flush()
    except Exception as e:
        print(f"\nError communicating with Ollama: {e}")
        continue

    print("\n")

    chat_history.append({"role": "user", "content": question})

    if NO_INFO_PHRASE.lower() in full_response.lower():
        chat_history.append({"role": "assistant", "content": NO_INFO_PHRASE})
        continue

    chat_history.append({"role": "assistant", "content": full_response})

    print("---- SOURCES ----\n")
    seen_sources = []
    for i, doc in enumerate(valid_docs, start=1):
        source = doc.metadata.get("source", "")
        if source in seen_sources:
            continue
        seen_sources.append(source)

        title   = doc.metadata.get("title", "No title")
        url     = DOCS_URL + source.replace("\\", "/")
        preview = " ".join(doc.page_content.split())[:200]

        print(f"[Source {i}]")
        print(f"Title:   {title}")
        print(f"URL:     {url}")
        print(f"Preview: {preview}...")
        print()
    print()