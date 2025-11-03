# Super simple example usage
from together_socket import TogetherClient


def main():
    # Initialize TogetherClient with a model id, temperature, and empty
    # conversation history
    adapter = TogetherClient()

    adapter.add_system_message("You are a concise helpful assistant.")

    # The new TogetherClient does not support per-thread storage in this
    # wrapper, so we include the author in the message content if desired.
    while True:
        user_message = input("User: ")
        adapter.add_user_message(user_message)
        reply = adapter.generate_response()
        print("AI: ", reply)


if __name__ == "__main__":
    main()
