
def get_response(user_input: str) -> str:
    message = user_input.lower().strip()

    if message in ["hello", "hi", "hey"]:
        return "Hello! How can I help you?"

    elif message in ["how are you", "how are you?"]:
        return "I'm doing great! Thanks for asking."

    elif message == "what is rag":
        return (
            "RAG stands for Retrieval-Augmented Generation. "
            "It retrieves relevant information before generating an answer."
        )

    elif message == "help":
        return (
            "Available commands:\n"
            "- hello: Greet the chatbot\n"
            "- how are you: Ask how the bot is doing\n"
            "- what is rag: Learn about RAG\n"
            "- help: Show available commands\n"
            "- bye or exit: End the conversation"
        )

    elif message in ["bye", "exit", "quit"]:
        return "Goodbye! See you next time."

    else:
        return "Sorry, I don't understand that yet. Type 'help' for commands."


def main():
    print("PyBot: Hi! I'm PyBot.")
    print("PyBot: Type 'help' for commands.")

    while True:
        user_input = input("You: ")

        response = get_response(user_input)
        print("PyBot:", response)

        if user_input.lower().strip() in ["bye", "exit", "quit"]:
            break


if __name__ == "__main__":
    main()
