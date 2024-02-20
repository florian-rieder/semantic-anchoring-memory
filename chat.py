"""Main conversation chain, used by main.py and cli.py to provide the
chat model with long term memory"""

from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.memory import (
    CombinedMemory, ConversationBufferWindowMemory
)

from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from memory.semantic.memory import SemanticLongTermMemory
from memory.semantic.store import SemanticStore, ABox, TBox
from memory.landwehr.landwehr import LandwehrMemory

from config import (
    ONTOLOGIES_PATHS,
    BASE_KNOWLEDGE_PATH,
    MEMORY_PATH,
    CLASS_DB_PATH,
    PREDICATES_DB_PATH,
    ENTITIES_DB_PATH,
    WORKHORSE_MODEL_NAME,
    CHAT_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    K_PREDICATES_TO_RETRIEVE,
    K_CLASSES_TO_RETRIEVE
)


# Longer system prompt supposed to be similar to ChatGPT's
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

CONVERSATION_PROMPT = PromptTemplate(
    input_variables=["history", "long_term_memory", "input"],
    template=template
)


def get_chain(stream_handler, memory_model = 'semantic') -> ConversationChain:
    """
    Create a streaming ConversationChain for question/answering.

    Parameters
    ----------
    stream_handler: The stream handler for processing the streaming
        responses.
    memory_model (optional): The memory model to use. Can be either
        "semantic" or "landwehr". Defaults to "semantic".

    Returns
    -------
    ConversationChain: The created ConversationChain object.

    Raises
    ------
    ValueError: If an unexpected memory model is provided.

    Summary
    -------
    The function creates a ConversationChain object for question/answering.
    It sets up the necessary components such as the streaming ChatOpenAI
    model, the workhorse ChatOpenAI model, the OpenAIEmbeddings for
    vector stores, and the conversation memory.

    If the memory_model is set to "semantic", it also creates a
    SemanticStore object and a SemanticLongTermMemory object for
    integrating with an external knowledge graph. The SemanticStore is
    initialized with the necessary parameters such as the encoder_llm,
    the TBox, and the ABox. The SemanticLongTermMemory is initialized
    with the background_llm and the semantic_store.

    If the memory_model is set to "landwehr", it creates a LandwehrMemory
    object with the background_llm.

    The ConversationChain is then created with the stream_llm, the
    CONVERSATION_PROMPT, the combined memory consisting of the
    conversation_memory and the long_term_memory, and the
    callback_manager for streaming. The verbose parameter is set to True.

    The created ConversationChain object is returned.
    """

    # Used for streaming
    manager = AsyncCallbackManager([])
    stream_manager = AsyncCallbackManager([stream_handler])

    # ChatLLM whose responses are streamed to the client
    stream_llm = ChatOpenAI(
        model=CHAT_MODEL_NAME,
        temperature=0.05,
        streaming=True,
        callback_manager=stream_manager
    )

    # Workhorse LLM
    background_llm = ChatOpenAI(
        model=WORKHORSE_MODEL_NAME,
        temperature=0
    )

    # Embedding function to be used by vector stores
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        show_progress_bar=False
    )

    # Regular conversation window memory
    conversation_memory = ConversationBufferWindowMemory(
        k=4,
        human_prefix='User',
        ai_prefix='Assistant',
        input_key='input'
    )

    if memory_model == 'semantic':
        semantic_store = SemanticStore(
            encoder_llm=background_llm,
            tbox=TBox(
                ontologies_paths=ONTOLOGIES_PATHS,
                predicates_db=Chroma(
                    persist_directory=PREDICATES_DB_PATH,
                    embedding_function=embeddings
                ),
                classes_db=Chroma(
                    persist_directory=CLASS_DB_PATH,
                    embedding_function=embeddings
                ),
            ),
            abox=ABox(
                entities_store=Chroma(
                    persist_directory=ENTITIES_DB_PATH,
                    embedding_function=embeddings
                ),
                memory_base_path=BASE_KNOWLEDGE_PATH,
                memory_path=MEMORY_PATH
            ),
            k_similar_classes=K_CLASSES_TO_RETRIEVE,
            k_similar_predicates=K_PREDICATES_TO_RETRIEVE
        )

        long_term_memory = SemanticLongTermMemory(
            llm=background_llm,
            semantic_store=semantic_store,
            k=4,
            human_prefix='User',
            ai_prefix='Assistant',
            memory_key='long_term_memory',
            input_key='input'
        )

    elif memory_model == 'landwehr':
        long_term_memory = LandwehrMemory(
            llm=background_llm,
            db=Chroma(
                persist_directory='./database/_memories/landwehr_memories_db',
                embedding_function=OpenAIEmbeddings(
                    model=EMBEDDING_MODEL_NAME
                )
            ),
            k=8,
            human_prefix='User',
            ai_prefix='Assistant',
            memory_key='long_term_memory',
            input_key='input'
        )
    else:
        raise ValueError(
            f'Unexpected memory model: {memory_model}."'
            ' Expected "semantic" or "landwehr".')

    conversation = ConversationChain(
        llm=stream_llm,
        prompt=CONVERSATION_PROMPT,
        memory=CombinedMemory(
            memories=[
                conversation_memory,
                long_term_memory
            ]
        ),
        callback_manager=manager,  # used for streaming
        verbose=True
    )

    return conversation
