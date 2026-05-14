import ollama

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)

print("Programming RAG Chatbot")
print("Type 'exit' to quit\n")

while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    docs = db.similarity_search(question, k=3)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a strict programming assistant.

You MUST answer ONLY using the provided documentation.

DO NOT use your own knowledge.
DO NOT make up answers.

If the answer is not clearly in the documentation, respond ONLY with:
"I don't know based on the provided documentation."

Documentation:
----------------
{context}
----------------

Question:
{question}

Answer:
"""

    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    print("\nBot:")
    print(answer)
    
    if "I don't know based on the provided documentation." not in answer:

        print("\n---- SOURCES ----\n")

        for i, doc in enumerate(docs):

            source = doc.metadata.get("source", "Unknown source")

            print(f"[Source {i+1}]")

            print(f"File: {source}")

            preview = doc.page_content[:300].replace("\n", " ")

            print(f"Text preview: {preview}")

            print()

    print()

