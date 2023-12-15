from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory, ConversationKGMemory
# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     HumanMessagePromptTemplate,
#     SystemMessagePromptTemplate,
# )
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.llms import FakeListLLM

from prompts import KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT, ENTITY_EXTRACTION_PROMPT

# template = (
# """Assistant is a large language model (LLM).

# Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand. If Assistant does not know the answer to a question, it truthfully says it does not know.

# Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

# Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

# Relevant information that Assistant learned from previous conversations:

# {history}

# User: {input}
# Assistant:
# """
# )

# Short system prompt to economize tokens
template = """The following is a friendly conversation between a human and an AI Assistant. Assistant is talkative and provides lots of specific details from its context. If Assistant does not know the answer to a question, it truthfully says it does not know.

Relevant information (for reference only):
{history}
User: {input}
Assistant:"""


PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=template)

llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

conversation = ConversationChain(
    #llm=FakeListLLM(responses=["Hello, I'm Assistant", "That's fucking great man", "And I like onions !"]),
    llm=llm,
    prompt=PROMPT,
    memory=ConversationKGMemory(
        llm=llm,
        k=8,
        human_prefix='User',
        ai_prefix='Assistant',
        knowledge_extraction_prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
        entity_extraction_prompt=ENTITY_EXTRACTION_PROMPT
    ),
    verbose=True,
    # memory=ConversationBufferWindowMemory(
    #     k=4,
    #     ai_prefix="Assistant",
    #     human_prefix="User"
    # )
)

#conversation.memory.save_context({"input": "hi"}, {"output": "whats up"})
# memory.save_context({"input": "say hi to john"}, {"output": "john! Who"})
# memory.save_context({"input": "he is a friend"}, {"output": "sure"})

# Define the chat function
def chat_with_chatbot(user_input):
    print(conversation.memory.load_memory_variables({'input': user_input}))
    # get a chat completion from the formatted messages
    response = conversation.predict(input=user_input)

    print(conversation.memory.kg.get_triples())
    # print(conversation.memory.chat_memory.messages)
    # #memory.load_memory_variables({"input": "who is john"})
    print(conversation.memory.load_memory_variables({'input': user_input}))

    # print(response)
    return response


if __name__ == '__main__':
    # Run the chat loop
    while True:
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        response = chat_with_chatbot(user_input)
        print("Assistant: " + response)
