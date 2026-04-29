import ollama

print("Programming Assistant Chatbot")
print("Type 'exit' to quit\n")

while True:

    # unos korisnika
    question = input("You: ")

    # izlaz iz programa
    if question.lower() == "exit":
        print("Goodbye!")
        break

    # slanje pitanja modelu
    response = ollama.chat(
        model="phi3",
        messages=[
            {
                "role": "user",
                "content": question
            }
        ]
    )

    # uzimanje odgovora
    answer = response["message"]["content"]

    print("\nBot:")
    print(answer)
    print()