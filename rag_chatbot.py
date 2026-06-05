import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

VECTOR_DB_PATH = "vector_db"
MODEL_NAME     = "phi3"
TOP_K          = 4   
DOCS_URL       = "https://docs.python.org/3/"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

print("Python RAG Chatbot")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    results = db.similarity_search_with_score(question, k=TOP_K)
    docs = [doc for doc, score in results]

    context_parts = []

    for i, doc in enumerate(docs, start=1):
        source_label = doc.metadata.get("title", "Unknown")
        context_parts.append(f"[Source {i}: {source_label}]\n{doc.page_content}")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an assistant that helps with Python programming.

Answer ONLY based on the provided documentation.
Do NOT use your own knowledge.
Do NOT make up answers.

If the answer is not in the documentation, respond exactly with:
"I don't have enough information in the documentation to answer this question."

When answering, you can reference sources by their numbers in square brackets,
e.g: "The os.path.join() function [Source 2] joins parts of a path..."

Documentation:
{context}

Question: {question}

Answer:"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    print("\nBot:")
    print(answer)

    print("\n---- SOURCES ----\n")

    seen_sources = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "")

        if source in seen_sources:
            continue
        seen_sources.append(source)

        title   = doc.metadata.get("title", "No title")
        section = doc.metadata.get("section", "general")
        url     = DOCS_URL + source.replace("\\", "/")
        preview = " ".join(doc.page_content.split())[:300]

        print(f"[Source {i}]")
        print(f"Title:   {title}")
        print(f"Section: {section}")
        print(f"URL:     {url}")
        print(f"Preview: {preview}...")
        print()

    print()