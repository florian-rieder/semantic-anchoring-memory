import argparse
import traceback

from langchain_community.vectorstores import Chroma

from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings

from memory.semantic.store import SemanticStore, TBox, ABox
from memory.semantic.learn import memorize


from config import (
    ONTOLOGIES_PATHS,
    CLASS_DB_PATH,
    PREDICATES_DB_PATH,
    ENTITIES_DB_PATH
)

def init():
    print('Initializing LLM...')
    llm = OpenAI(
        model='gpt-3.5-turbo-instruct',
    )

    print('Initializing embeddings...')
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
    )


    print('Initializing T-Box...')
    tbox = TBox(ONTOLOGIES_PATHS)
    abox = ABox(
        entities_store=Chroma(
            persist_directory=ENTITIES_DB_PATH,
            embedding_function=embeddings
        )
    )

    print('Initializing vector stores...')
    store = SemanticStore(
        predicates_db=Chroma(
            persist_directory=PREDICATES_DB_PATH,
            embedding_function=embeddings
        ),
        classes_db=Chroma(
            persist_directory=CLASS_DB_PATH,
            embedding_function=embeddings
        ),
        encoder_llm=llm,
        tbox=tbox,
        abox=abox
    )

    return llm, store


def ingest_files(files):
    llm, store = init()

    for file in files:
        try:
            with open(file, 'r') as f:
                content = f.read()
        except Exception:
            traceback.print_exc()
            continue

        memorize(content, llm, store)


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process one or more files.')

    # Add the file argument
    parser.add_argument('-f', '--files', nargs='+', help='List of files to process')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Ingest the list of files
    ingest_files(args.files)
