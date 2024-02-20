"""
This module provides a command line interface for the chat application
that uses a long term memory model. 
The memory model can be either 'landwehr' or 'semantic', with 'semantic'
being the default.

Arguments
---------
-m, --memory: the memory model to use. Can be either 'landwehr' or
    'semantic', with 'semantic' being the default.

Usage
-----
>>> python cli.py [-m {landwehr,semantic}]

Example
-------
>>> python cli.py -m landwehr

This will start the chat application with the 'landwehr' memory model. 
To end the conversation and memorize it, type 'exit'.
"""

import argparse

from langchain.chains import ConversationChain
from langchain.callbacks import StdOutCallbackHandler

from chat import get_chain


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
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Memory Chat's command line interface")
    parser.add_argument('-m', '--memory',
                        choices=['landwehr', 'semantic'],
                        default='semantic',
                        help="Choose memory model: 'landwehr' or 'semantic'")
    args = parser.parse_args()

    stream_handler = StdOutCallbackHandler()
    conversation = get_chain(stream_handler, memory_model=args.memory)

    print("Welcome to Memory Chat's command line interface !"
          " Type your message to start chatting."
          " Type 'exit' to end the conversation and memorize.")

    # Run the chat loop
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            end_conversation(conversation)
            break
        response = conversation.predict(input=user_input)
        print("Assistant: " + response)
