from langchain.chat_models import ChatOpenAI
from langchain_community.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.memory import (
    CombinedMemory,
    ConversationBufferWindowMemory
)

# from memory import SemanticLongTermMemory
from server.memory.landwehr import LandwehrMemory


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
template = """The following is a friendly conversation between a human and an AI Assistant.
Assistant is talkative and provides lots of specific details from its context. Assistant is also able to learn and recall facts from previous conversations.
If Assistant does not know the answer to a question, it truthfully says it does not know.
If the user asks about memories, and nothing relevant is in the relevant information section, Assistant truthfully says it doesn't remember.

Relevant information (for reference only):
{long_term_memory}

Conversation history:
{history}

User: {input}
Assistant:"""


PROMPT = PromptTemplate(
    input_variables=["history", "long_term_memory", "input"],
    template=template
)


def get_chain(stream_handler) -> ConversationChain:
    """Create a streaming ConversationChain for question/answering."""

    # Used for streaming
    manager = AsyncCallbackManager([])
    stream_manager = AsyncCallbackManager([stream_handler])

    stream_llm = ChatOpenAI(
        model='gpt-3.5-turbo',
        temperature=0.05,
        streaming=True,
        callback_manager=stream_manager
    )

    background_llm = OpenAI(
        model='gpt-3.5-turbo-instruct',
        temperature=0
    )

    # Regular conversation window memory
    conversation_memory = ConversationBufferWindowMemory(
        k=4,
        human_prefix='User',
        ai_prefix='Assistant',
        input_key='input'
    )

    # long_term_memory = SemanticLongTermMemory(
    #     llm=llm,
    #     k=4,
    #     human_prefix='User',
    #     ai_prefix='Assistant',
    #     memory_key='long_term_memory',
    #     input_key='input'
    # )

    long_term_memory = LandwehrMemory(
        llm=background_llm,
        db=Chroma(
            persist_directory='./database/_memories/landwehr_memories_db',
            embedding_function=OpenAIEmbeddings(
                model='text-embedding-ada-002'
            )
        ),
        k=8,
        human_prefix='User',
        ai_prefix='Assistant',
        memory_key='long_term_memory',
        input_key='input'
    )

    conversation = ConversationChain(
        llm=stream_llm,
        prompt=PROMPT,
        memory=CombinedMemory(
            memories=[
                conversation_memory,
                long_term_memory
            ]
        ),
        callback_manager=manager, # used for streaming
        verbose=True
    )

    return conversation
