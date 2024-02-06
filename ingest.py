import argparse
import traceback
from tqdm import tqdm

from langchain_community.vectorstores import Chroma

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from memory.semantic.store import SemanticStore, TBox, ABox
from memory.semantic.learn import memorize


from config import (
    ONTOLOGIES_PATHS,
    MEMORY_PATH,
    BASE_KNOWLEDGE_PATH,
    CLASS_DB_PATH,
    PREDICATES_DB_PATH,
    ENTITIES_DB_PATH
)

def init(memory_path: str, base_knowledge: str):
    print('Initializing LLM...', end=' ', flush=True)
    llm = ChatOpenAI(
        model='gpt-3.5-turbo-1106',
        temperature=0
    )
    print('Done.')

    print('Initializing embeddings...', end=' ', flush=True)
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
    )
    print('Done.')

    print('Initializing T-Box...', end=' ', flush=True)
    tbox = TBox(
        ontologies_paths=ONTOLOGIES_PATHS,
        predicates_db=Chroma(
            persist_directory=PREDICATES_DB_PATH,
            embedding_function=embeddings
        ),
        classes_db=Chroma(
            persist_directory=CLASS_DB_PATH,
            embedding_function=embeddings
        )
    )
    print('Done.')

    print('Initializing A-Box...', end=' ', flush=True)
    abox = ABox(
        entities_store=Chroma(
            persist_directory=ENTITIES_DB_PATH,
            embedding_function=embeddings
        ),
        memory_base_path=base_knowledge,
        memory_path=memory_path
    )
    print('Done.')


    print('Initializing Semantic Store...', end=' ', flush=True)
    store = SemanticStore(
        encoder_llm=llm,
        tbox=tbox,
        abox=abox
    )
    print('Done.')

    return llm, store


def ingest_files(files, llm, store):

    for file in tqdm(files):
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

    # Add the files argument
    parser.add_argument('-f', '--files',
                        dest='files',
                        nargs='+',
                        help='List of files to process')
    # Add the MEMORY_PATH argument
    parser.add_argument('-o', '--output',
                        dest='memory_path',
                        help='Path to save the generated graph to (memory path)',
                        default=MEMORY_PATH)
    # Add the BASE_KNOWLEDGE_PATH argument
    parser.add_argument('-b', '--base-knowledge',
                        dest='base_knowledge',
                        help='Path to preconceived knowledge file (base knowledge)',
                        default=BASE_KNOWLEDGE_PATH)

    # Parse the command-line arguments
    args = parser.parse_args()

    if args.base_knowledge == 'None':
        args.base_knowledge = None

    llm, store = init(args.memory_path, args.base_knowledge)

    # Ingest the list of files
    ingest_files(args.files, llm, store)
