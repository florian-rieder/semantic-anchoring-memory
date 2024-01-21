

from memory.semantic.store import SemanticStore
from memory.semantic.abox import ABox
from memory.semantic.tbox import TBox

from config import (
    ONTOLOGIES_PATHS,
    CLASS_DB_PATH,
    PREDICATES_DB_PATH,
    ENTITIES_DB_PATH
)


def generate_tbox_db(store: SemanticStore):
    """Generate the T-Box vector database (databases containing all the
    classes and predicates from the ontologies specified in the
    SemanticStore)
    """
    # Load the classes and predicates into vector stores

    print('Loading classes...')
    classes = store.tbox.get_classes_embedding_strings()
    print(f'Number of classes: {len(classes)}')
    with open('ontologies/classes.owl', 'w') as f:
        for c in classes:
            f.write(c + '\n')

    print('Loading predicates...')
    predicates = store.tbox.get_predicates_embedding_strings()
    print(f'Number of predicates: {len(predicates)}')
    with open('ontologies/predicates.owl', 'w') as f:
        for p in predicates:
            f.write(p + '\n')

    # Storage into vector databases
    print('Storing classes...')
    store.store_classes(classes)

    print('Storing predicates...')
    store.store_predicates(predicates)


if __name__ == '__main__':
    from langchain_community.llms import FakeListLLM
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    print('Initializing embeddings...')
    # Embedding function to be used by vector stores
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        show_progress_bar=False
    )

    print('Initializing T-Box...')
    tbox = TBox(ONTOLOGIES_PATHS)

    print('Initializing A-Box...')
    abox = ABox(
        entities_store=Chroma(
            persist_directory=ENTITIES_DB_PATH,
            embedding_function=embeddings
        )
    )

    print('Initializing SemanticStore...')
    # Semantic store we're going to use
    store = SemanticStore(
        predicates_db=Chroma(
            persist_directory=PREDICATES_DB_PATH,
            embedding_function=embeddings
        ),
        classes_db=Chroma(
            persist_directory=CLASS_DB_PATH,
            embedding_function=embeddings
        ),
        # We don't need an LLM to generate the T-Box DBs
        encoder_llm=FakeListLLM(responses=[]),
        tbox=tbox,
        abox=abox
    )

    print('Generating T-Box databases...')
    # Generate the classes and predicates vector databases
    generate_tbox_db(store)
