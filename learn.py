"""
Responsibility:
    Converts raw text into condensed memories for virtual AI characters.
Process:
    Splits the source text into smaller chunks, processes each chunk
    using the LLM to extract important facts, and creates corresponding
    memories.
"""
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate


from prompts import (
    FACT_EXTRACTION_PROMPT,
    NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
)


def split_chunk_context_pairs(text: str, llm: BaseLanguageModel) -> List[tuple]:
    """Split text into multiple components."""
    # 1. Split source text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # 2. Process each chunk in isolation but with a context that contains a
    # summary of all previous chunks.
    contexts = []
    for idx, chunk in enumerate(chunks):
        chunks_before = chunks[:idx]

        context = "".join(chunks_before)

        summary = summarize(context, llm)

        context.append((chunk, summary))

    return contexts


def summarize(text: str, llm: BaseLanguageModel) -> str:
    # Define summarizer prompt
    summarizer_prompt_template = (
        "Generate a concise summary of the conversation transcript, focusing on key"
        " facts and memorable details related to the user's life."
        " Highlight significant events, achievements, personal preferences, and any"
        " noteworthy information that provides a comprehensive overview of the user's experiences and interests:\n\n"
        "Conversation history:\n\n"
        "{text}"
        "\n\nSummary of the transcript:"
    )
    summarizer_prompt = PromptTemplate(
        input_variables=['text'],
        template=summarizer_prompt_template,
    )

    chain = LLMChain(
        llm=llm,
        prompt=summarizer_prompt,
    )

    response = chain.predict(
        text=text
    )
    # TODO: Output validation and parsing

    return response


def extract_facts(chunk: str, summary: str, llm: BaseLanguageModel) -> List[str]:
    # Define fact extraction prompt
    chain = LLMChain(
        llm=llm,
        prompt=FACT_EXTRACTION_PROMPT
    )

    response = chain.predict(chunk=chunk, summary=summary)

    # TODO: Parse output

    return response


def extract_triplets(summary: str, llm: BaseLanguageModel):
    chain = LLMChain(
        # prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
        prompt=NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
        llm=llm,
        # verbose=True
    )

    triplets = chain.predict(summary=summary)

    return triplets


if __name__ == "__main__":
    from langchain.llms import OpenAI

    with open('_work/example_conversation3.txt', 'r') as f:
        test_text = f.read()

    llm = OpenAI(
        model='gpt-3.5-turbo-instruct',
        temperature=0,
        # frequency_penalty=0.2
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2048, chunk_overlap=128)
    chunks = text_splitter.split_text(test_text)

    triplets = []
    for chunk in chunks:
        summary = summarize(chunk, llm)
        summary_sentences = [s.strip() for s in summary.split('. ')]

        for sentence in summary_sentences:
            sentence_triplets = extract_triplets(sentence, llm)
            triplets.append(sentence_triplets)

    print(triplets)
