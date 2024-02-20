"""
This module is used to ingest the content of multiple files into a
Semantic Store. 

It initializes the language model, embeddings, T-Box, A-Box, and
Semantic Store, and then ingests the content of the files.

Usage:
------
python ingest.py -f file1.txt file2.txt -o output_path -b base_knowledge_path

Arguments
---------
-f, --files: List of files to process. Provide the file names separated
    by space.
-o, --output: Path to save the generated graph to (memory path). If not
    provided, it defaults to the value of MEMORY_PATH in config.py.
-b, --base-knowledge: Path to preconceived knowledge file (base
    knowledge). If not provided, it defaults to the value of
    BASE_KNOWLEDGE_PATH in config.py. If you want to specify no base
    knowledge, use 'None'.

Example
-------
python filename.py -f file1.txt file2.txt -o /path/to/memory -b /path/to/base/knowledge
"""

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
    ENTITIES_DB_PATH,
    WORKHORSE_MODEL_NAME,
    EMBEDDING_MODEL_NAME,
    K_CLASSES_TO_RETRIEVE,
    K_PREDICATES_TO_RETRIEVE
)


def init(memory_path: str,
         base_knowledge: str,
         ontologies_paths: str = ONTOLOGIES_PATHS,
         predicates_db_path: str = PREDICATES_DB_PATH,
         class_db_path: str = CLASS_DB_PATH,
         entities_db_path: str = ENTITIES_DB_PATH
         ):
    """
    Initializes the language model, embeddings, T-Box, A-Box, and
    Semantic Store.

    Parameters
    ----------
    memory_path (str): The path to the memory directory.
    base_knowledge (str): The path to the base knowledge directory.
    ontologies_paths (str, optional): The paths to the ontologies. Defaults to ONTOLOGIES_PATHS.
    predicates_db_path (str, optional): The path to the predicates database. Defaults to PREDICATES_DB_PATH.
    class_db_path (str, optional): The path to the classes database. Defaults to CLASS_DB_PATH.
    entities_db_path (str, optional): The path to the entities database. Defaults to ENTITIES_DB_PATH.

    Returns
    -------
    llm (ChatOpenAI): The initialized language model.
    store (SemanticStore): The initialized Semantic Store.
    """

    print('Initializing LLM...', end=' ', flush=True)
    llm = ChatOpenAI(
        model=WORKHORSE_MODEL_NAME,
        temperature=0
    )
    print('Done.')

    print('Initializing embeddings...', end=' ', flush=True)
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
    )
    print('Done.')

    print('Initializing T-Box...', end=' ', flush=True)
    tbox = TBox(
        ontologies_paths=ontologies_paths,
        predicates_db=Chroma(
            persist_directory=predicates_db_path,
            embedding_function=embeddings
        ),
        classes_db=Chroma(
            persist_directory=class_db_path,
            embedding_function=embeddings
        )
    )
    print('Done.')

    print('Initializing A-Box...', end=' ', flush=True)
    abox = ABox(
        entities_store=Chroma(
            persist_directory=entities_db_path,
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
        abox=abox,
        k_similar_classes=K_CLASSES_TO_RETRIEVE,
        k_similar_predicates=K_PREDICATES_TO_RETRIEVE
    )
    print('Done.')

    return llm, store


def ingest_files(files, llm, store):
    """
    Ingests the content of multiple files into the Semantic Store.
    Outputs files as defined in config.py

    Parameters
    ----------
    files : list
        A list of file paths to be ingested.
    llm : ChatOpenAI
        The initialized language model.
    store : SemanticStore
        The initialized Semantic Store.
    """

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
