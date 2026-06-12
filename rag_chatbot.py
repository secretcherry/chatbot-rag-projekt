import sys
import ollama
import config
import rag_engine

def main():
    print("Python RAG Chatbot")
    print("Type 'exit' to quit\n")

    chat_history = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if question.lower() == "exit":
            break
        if not question:
            continue

        search_query = question
        if chat_history:
            print("Thinking... understanding context")
            try:
                search_query = rag_engine.get_standalone_question(chat_history, question)
            except Exception as e:
                print(f"Error connecting to Ollama: {e}")
                continue

        print(f'Searching database for: "{search_query}"')

        valid_docs = rag_engine.retrieve_documents(search_query, question)

        if not valid_docs:
            print(f"\nBot:\n{config.NO_INFO_PHRASE}\n")
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": config.NO_INFO_PHRASE})
            continue

        context_parts = []
        for i, doc in enumerate(valid_docs, start=1):
            title = doc.metadata.get("title", "Unknown")
            context_parts.append(f"[Source {i}: {title}]\n{doc.page_content}")

        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": rag_engine.get_system_prompt()},
            {"role": "user",   "content": f"Documentation Context:\n{context}\n\nQuestion: {search_query}\n\nAnswer (use ONLY the context above):"},
        ]

        print("\nBot:")
        try:
            response_stream = ollama.chat(model=config.MODEL_NAME, messages=messages, stream=True)
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

        if config.NO_INFO_PHRASE.lower() in full_response.lower():
            chat_history.append({"role": "assistant", "content": config.NO_INFO_PHRASE})
            continue

        chat_history.append({"role": "assistant", "content": full_response})

        print("---- SOURCES ----\n")
        sources = rag_engine.format_sources(valid_docs)
        for i, src in enumerate(sources, start=1):
            print(f"[Source {i}]")
            print(f"Title:   {src['title']}")
            print(f"URL:     {src['url']}")
            print(f"Preview: {src['preview']}...")
            print()
        print()

if __name__ == "__main__":
    main()