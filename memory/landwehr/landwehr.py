from typing import Any, Dict, List
from tqdm import tqdm

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models import BaseLanguageModel
from langchain_core.vectorstores import VectorStore
from langchain.memory.chat_memory import BaseChatMemory

from memory.prompts import (
    FACT_EXTRACTION_PROMPT, QUERY_CREATION_PROMPT
)


class LandwehrMemory(BaseChatMemory):
    """
    Memory class for storing memories, following the architecture
    laid out in Landwehr et al. (2023)
    """

    # Define key to pass information about entities into prompt.
    memory_key: str = "memories"

    llm: BaseLanguageModel
    db: VectorStore

    @property
    def memory_variables(self) -> List[str]:
        """Define the variables we are providing to the prompt."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Load the memory variables, in this case the memories from the vector database."""
        # Generate a query for the memories
        query = generate_retrieval_query(inputs['input'], self.llm)
        print(query)

        # Do a similarity search on the memories
        results = [d.page_content for d in self.db.similarity_search(query)]
        print(results)
        output_string = "\n\n".join(results)

        # Return combined information about entities to put into context.
        return {self.memory_key: output_string}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        # Actually, do nothing. Our save_context part is at the end of a
        # conversation, not at the end of a conversation turn.
        # We'll save new info in memorize(), which is called externally, when
        # the conversation has ended.
        pass

    def memorize(self, conversation: str):
        """Function to run a given text in the memory creation pipeline"""
        memorize(conversation, self.llm, self.db)

    def clear(self):
        # os.rmdir(self.db.persist_directory) # Untested
        pass


def memorize(input_text: str, llm: BaseLanguageModel, store: VectorStore):
    chunk_summary_pairs = split_chunk_context_pairs(input_text, llm)

    extracted_facts = []
    print('Extracting facts from (chunk, summary) pairs...')
    for chunk, summary in tqdm(chunk_summary_pairs):
        facts = extract_facts(chunk, summary, llm)
        print(facts)
        extracted_facts += facts

    # TODO: These facts are then post-processed by resolving references
    # (e.g., pronouns), ensuring that each fact is understandable atomically
    # (without context)

    print('Adding facts to memory...')
    store.add_texts(extracted_facts)


def split_chunk_context_pairs(text: str, llm: BaseLanguageModel, chunk_size=2048) -> List[tuple]:
    """Split text into multiple chunks, and for each chunk create a summary
    that takes into account previous chunks."""
    # 1. Split source text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # 2. Process each chunk in isolation but with a context that contains a
    # summary of all previous chunks.
    contexts = []
    for idx, chunk in enumerate(chunks):
        chunks_before = chunks[:idx+1]
        context = "".join(chunks_before)

        summary = summarize(context, llm)
        contexts.append((chunk, summary))
        break

    return contexts


def summarize(text: str, llm: BaseLanguageModel) -> str:
    # Define summarizer prompt
    summarizer_prompt_template = (
        "Generate a concise summary of the conversation transcript, focusing on key"
        " facts and memorable details related to the user's life."
        " Write sentences which are understandable in isolation. Always refer to named entities by their name. Prioritise proper name over generic names."
        " Always refer to the user as User, and to the AI as Assistant."
        " Highlight significant events, achievements, personal preferences, and any"
        " noteworthy information that provides a comprehensive overview of the user's experiences and interests:\n\n"
        "Conversation history:\n\n"
        "{text}"
        "\n\nSummary of the transcript:\n"
    )
    summarizer_prompt = PromptTemplate(
        input_variables=['text'],
        template=summarizer_prompt_template,
    )

    chain = LLMChain(
        llm=llm,
        prompt=summarizer_prompt,
        verbose=True
    )

    response = chain.predict(
        text=text
    )
    # TODO: Output validation and parsing

    return response


def summarize_chunk(summary_of_previous_chunks, chunk, llm):
    chunk_summarizer_prompt_template = (
        'Generate a concise summary of the given chunk of conversation transcript, focusing on key'
        " facts and memorable details related to the user's life and the conversation topic."
        'Summary of previous chunks (for reference only):\n'
        f'{summary_of_previous_chunks}'
        'Chunk to summarize'
        f'{chunk}'
    )

    summarizer_prompt = PromptTemplate(
        input_variables=['summary_of_previous_chunks', 'chunk'],
        template=chunk_summarizer_prompt_template,
    )

    chain = LLMChain(
        llm=llm,
        prompt=summarizer_prompt
    )

    response = chain.predict(
        summary_of_previous_chunks=summary_of_previous_chunks,
        chunk=chunk
    )
    return response


def extract_facts(chunk: str, summary: str, llm: BaseLanguageModel) -> List[str]:
    # Define fact extraction prompt
    chain = LLMChain(
        llm=llm,
        prompt=FACT_EXTRACTION_PROMPT
    )

    response = chain.predict(chunk=chunk, summary=summary)

    # TODO: Parse output
    facts = [f.strip().strip('-').strip()
             for f in response.strip().split('\n')]

    return facts


def generate_retrieval_query(input_query: str, llm: BaseLanguageModel) -> str:
    chain = LLMChain(
        llm=llm,
        prompt=QUERY_CREATION_PROMPT,
        verbose=True,
    )

    refined_query = chain.predict(history=input_query)

    return refined_query
