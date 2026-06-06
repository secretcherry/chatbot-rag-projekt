import streamlit as st
import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# ── Config ───────────────────────────────────────────────────────────────────
VECTOR_DB_PATH = "vector_db"
MODEL_NAME     = "qwen2.5:7b"
TOP_K          = 8
DOCS_URL       = "https://docs.python.org/3/"
NO_INFO_PHRASE = "I don't have enough information in the documentation to answer this question."

HARD_THRESHOLD = 1.05
EASY_THRESHOLD = 0.65
RERANK_CUTOFF  = 2.0

# ── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_db():
    emb = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return FAISS.load_local(VECTOR_DB_PATH, emb, allow_dangerous_deserialization=True)

@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

db       = load_db()
reranker = load_reranker()

# ── RAG helpers ───────────────────────────────────────────────────────────────
def is_clearly_offtopic(q: str) -> bool:
    off_topic = [
        "what time", "what is the time", "current time",
        "what is a cat", "what is a dog", "what is a bird", "what is an animal",
        "weather", "recipe", "cook", "sport", "movie", "music", "football",
        "who is", "where is", "when was",
    ]
    return any(kw in q.lower() for kw in off_topic)

def check_relevance_with_llm(question: str) -> bool:
    prompt = (
        "Is the following question related to Python programming, Python syntax, "
        "Python libraries, or Python documentation?\n"
        "Answer with exactly one word: YES or NO.\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    r = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
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
    r = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    rewritten = r["message"]["content"].strip().splitlines()[0].strip().strip("'\"")
    return rewritten if len(rewritten.split()) <= 15 else current_q

def search_and_answer(question: str, history: list) -> dict:
    if is_clearly_offtopic(question):
        return {"answer": NO_INFO_PHRASE, "sources": [], "search_query": question, "blocked": True}

    search_query = question
    if history:
        search_query = get_standalone_question(history, question)

    faiss_results = db.similarity_search_with_score(search_query, k=TOP_K)
    if not faiss_results:
        return {"answer": NO_INFO_PHRASE, "sources": [], "search_query": search_query, "blocked": True}

    best_faiss_score = faiss_results[0][1]

    if best_faiss_score > HARD_THRESHOLD:
        return {"answer": NO_INFO_PHRASE, "sources": [], "search_query": search_query, "blocked": True}
    elif best_faiss_score > EASY_THRESHOLD:
        if not check_relevance_with_llm(question):
            return {"answer": NO_INFO_PHRASE, "sources": [], "search_query": search_query, "blocked": True}

    docs_only = [doc for doc, _ in faiss_results]
    pairs     = [(search_query, doc.page_content) for doc in docs_only]
    rscores   = reranker.predict(pairs)
    ranked    = sorted(zip(docs_only, rscores), key=lambda x: x[1], reverse=True)
    valid_docs = [doc for doc, score in ranked if score > RERANK_CUTOFF][:3]

    if not valid_docs:
        return {"answer": NO_INFO_PHRASE, "sources": [], "search_query": search_query, "blocked": True}

    context = "\n\n---\n\n".join(
        f"[Source {i}: {doc.metadata.get('title','?')}]\n{doc.page_content}"
        for i, doc in enumerate(valid_docs, 1)
    )
    system_prompt = (
        "You are a Python documentation assistant.\n\n"
        "STRICT RULES:\n"
        "- Use ONLY information from the Documentation Context.\n"
        f"- If the answer is not in the context, reply exactly: {NO_INFO_PHRASE}\n"
        "- Keep answers short and factual."
    )
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Documentation Context:\n{context}\n\nQuestion: {search_query}\n\nAnswer:"},
        ],
    )
    answer = response["message"]["content"].strip()

    sources = []
    if NO_INFO_PHRASE.lower() not in answer.lower():
        seen = []
        for doc in valid_docs:
            src = doc.metadata.get("source", "")
            if src in seen:
                continue
            seen.append(src)
            sources.append({
                "title":   doc.metadata.get("title", "No title"),
                "url":     DOCS_URL + src.replace("\\", "/"),
                "preview": " ".join(doc.page_content.split())[:200],
            })

    return {"answer": answer, "sources": sources, "search_query": search_query, "blocked": False}


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PyDocs Chat", layout="wide")

# ── Session state ─────────────────────────────────────────────────────────────
if "sessions" not in st.session_state:
    st.session_state.sessions = {"New Chat": []}
if "current" not in st.session_state:
    st.session_state.current = "New Chat"
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

def current_history():
    return st.session_state.sessions.get(st.session_state.current, [])

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Conversations**")
    st.divider()

    for name in list(st.session_state.sessions.keys()):
        is_active = (name == st.session_state.current)
        label = f"> {name}" if is_active else name
        if st.button(label, key=f"sess_{name}", use_container_width=True):
            st.session_state.current = name
            st.rerun()

    st.divider()

    if st.button("+ New Chat", use_container_width=True):
        existing = list(st.session_state.sessions.keys())
        n = 1
        new_name = "New Chat"
        while new_name in existing:
            n += 1
            new_name = f"New Chat {n}"
        st.session_state.sessions[new_name] = []
        st.session_state.current = new_name
        st.rerun()

    # Delete current session
    if len(st.session_state.sessions) > 1:
        if st.button("Delete Chat", use_container_width=True):
            del st.session_state.sessions[st.session_state.current]
            st.session_state.current = list(st.session_state.sessions.keys())[0]
            st.session_state.input_key += 1
            st.rerun()

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("PyDocs Chat")
st.caption("Python 3 documentation assistant")
st.divider()

history = current_history()

if not history:
    st.write("Ask anything about Python 3.")
    st.caption("Try: `what is a class?` · `how do I use decorators?` · `what is a generator?`")
else:
    for msg in history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                query = msg.get("search_query", "")
                orig  = msg.get("original_question", query)
                if query and query.lower() != orig.lower():
                    st.caption(f"searched: {query}")

                st.markdown(msg["content"])

                sources = msg.get("sources", [])
                if sources:
                    st.divider()
                    st.caption("Sources")
                    for s in sources:
                        st.caption(f"**{s['title']}**  \n[{s['url']}]({s['url']})")
                        st.markdown(
                            f"<small style='color:gray'>{s['preview']}...</small>",
                            unsafe_allow_html=True
                        )

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about Python...")

if user_input and user_input.strip():
    question = user_input.strip()
    history.append({"role": "user", "content": question})

    with st.spinner("Searching..."):
        result = search_and_answer(question, history[:-1])

    history.append({
        "role":              "assistant",
        "content":           result["answer"],
        "sources":           result["sources"],
        "search_query":      result["search_query"],
        "original_question": question,
    })

    # Auto-rename session on first message
    if len(history) == 2 and st.session_state.current.startswith("New Chat"):
        short = question[:28] + ("..." if len(question) > 28 else "")
        sessions = dict(st.session_state.sessions)
        sessions[short] = sessions.pop(st.session_state.current)
        st.session_state.sessions = sessions
        st.session_state.current  = short

    st.rerun()