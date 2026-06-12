import streamlit as st
import ollama
import config
import rag_engine

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
                        st.caption(f"**{s['title']}**  \n[Read source & highlight]({s['url']})")
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
        try:
            search_query = question
            if len(history) > 1:
                search_query = rag_engine.get_standalone_question(history[:-1], question)

            valid_docs = rag_engine.retrieve_documents(search_query, question)
            
            if not valid_docs:
                answer = config.NO_INFO_PHRASE
                sources = []
            else:
                context = "\n\n---\n\n".join(
                    f"[Source {i}: {doc.metadata.get('title','?')}]\n{doc.page_content}"
                    for i, doc in enumerate(valid_docs, 1)
                )
                
                messages = [{"role": "system", "content": rag_engine.get_system_prompt()}]
                
                for msg in history[-4:-1]:
                    role = "user" if msg["role"] == "user" else "assistant"
                    content = msg.get("original_question", msg["content"]) if role == "user" else msg["content"]
                    messages.append({"role": role, "content": content})
                
                messages.append({
                    "role": "user", 
                    "content": f"Documentation Context:\n{context}\n\nQuestion: {search_query}\n\nAnswer:"
                })
                
                response = ollama.chat(
                    model=config.MODEL_NAME,
                    messages=messages,
                )
                answer = response["message"]["content"].strip()
                
                if config.NO_INFO_PHRASE.lower() in answer.lower():
                    sources = []
                else:
                    sources = rag_engine.format_sources(valid_docs)

            history.append({
                "role":              "assistant",
                "content":           answer,
                "sources":           sources,
                "search_query":      search_query,
                "original_question": question,
            })

            if len(history) == 2 and st.session_state.current.startswith("New Chat"):
                short = question[:28] + ("..." if len(question) > 28 else "")
                sessions = dict(st.session_state.sessions)
                sessions[short] = sessions.pop(st.session_state.current)
                st.session_state.sessions = sessions
                st.session_state.current  = short

            st.rerun()

        except Exception as e:
            st.error(f"Error connecting to models: {e}")
            if len(history) > 0:
                history.pop() 