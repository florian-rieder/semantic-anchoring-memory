from langchain.chains import ConversationChain

from server.chat import get_chain


def end_conversation(conversation: ConversationChain) -> None:
    """
    At the end of the conversation, pass the conversation history through
    the memory creation pipeline.
    """
    # Get the conversation history from the conversation memory
    chat_history = conversation.memory.memories[0].chat_memory
    print(chat_history)
    # Memorize the conversation
    conversation.memory.memories[1].memorize(str(chat_history))


if __name__ == '__main__':
    from langchain.callbacks import StdOutCallbackHandler

    stream_handler = StdOutCallbackHandler()
    conversation = get_chain(stream_handler)

    print("Welcome to Memory Chat's command line interface ! Type your message to start chatting.")
    # Run the chat loop
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            end_conversation(conversation)
            break
        response = conversation.predict(input=user_input)
        print("Assistant: " + response)