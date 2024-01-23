"""
Responsibility:
    Converts raw text into condensed memories for virtual AI characters.
Process:
    Splits the source text into smaller chunks, processes each chunk
    using the LLM to extract important facts, and creates corresponding
    memories.
"""
from typing import List
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


from memory.prompts import (
    NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
    TRIPLET_ENCODER_PROMPT
)
from memory.semantic.store import SemanticStore


def memorize(conversation_history: str, llm: BaseLanguageModel, semantic_store: SemanticStore):
    # Extract raw triplets from the conversation history
    raw_triplets = conversation_to_triplets(conversation_history, llm)
    print(list(raw_triplets))

    # Encode triplet to RDF
    encoded_triplets = semantic_store.encode_triplets(raw_triplets)
    print(list(encoded_triplets))

    # Add encoded triplets to graph
    semantic_store.memorize_encoded_triplets(encoded_triplets)


def parse_triplet_string(triplets_string: str) -> tuple[str, str, str]:
    """
    Parse a string containing multiple triplets into a list of tuples
    """
    TRIPLET_PARSING_PATTERN = re.compile(r'\((?:([^,]+?),)\s*(?:([^,]+?),)\s*([^,]+?)\)')
    # split triplets apart
    triplets = TRIPLET_PARSING_PATTERN.findall(triplets_string)

    return triplets


def split_chunk_context_pairs(text: str, llm: BaseLanguageModel) -> List[tuple]:
    """Split text into multiple chunks, and for each chunk create a summary
    that takes into account previous chunks."""

    # 1. Split source text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # 2. Process each chunk in isolation but with a context that contains a
    # summary of all previous chunks.
    contexts = []
    for idx, chunk in enumerate(chunks):
        chunks_before = chunks[:idx+1]
        context = "".join(chunks_before)

        # context = 'There are no previous chunks. The chunk to summarize is the start of the conversation.'
        # if idx > 0:
        #     context = contexts[idx-1][1]

        # summary = summarize_chunk(context, chunk, llm)
        summary = summarize(context, llm)
        contexts.append((chunk, summary))
        break

    return contexts


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
        #verbose=True
    )

    response = chain.predict(
        text=text
    )
    # TODO: Output validation and parsing

    return response


def extract_triplets(summary: str, llm: BaseLanguageModel) -> list[tuple[str, str, str]]:
    chain = LLMChain(
        # prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
        prompt=NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
        llm=llm,
        # verbose=True
    )

    triplets = chain.predict(summary=summary)

    return triplets


def conversation_to_triplets(conversation: str, llm: BaseLanguageModel):
    conversation_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3072, chunk_overlap=256)
    chunks = conversation_splitter.split_text(conversation)

    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=64)

    print(f'Number of chunks: {len(chunks)}')
    triplets = []
    for chunk in chunks:
        summary = summarize(chunk, llm)

        summary_sentences = summary_splitter.split_text(summary)
        print(
            f'Number of chunks in the summary chunk: {len(summary_sentences)}')

        for sentence in summary_sentences:
            sentence_triplets = extract_triplets(sentence, llm)
            list_sentence_triplets = parse_triplet_string(sentence_triplets)
            triplets += list_sentence_triplets

    return triplets
