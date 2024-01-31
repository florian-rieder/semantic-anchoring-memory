

from memory.semantic.store import SemanticStore, ABox, TBox

from config import (
    ONTOLOGIES_PATHS,
    BASE_KNOWLEDGE_PATH,
    MEMORY_PATH,
    CLASS_DB_PATH,
    PREDICATES_DB_PATH,
    ENTITIES_DB_PATH
)


def generate_tbox_db(tbox: TBox,
                     debug_classes_file='ontologies/classes_dbpedia.owl',
                     debug_predicates_file='ontologies/predicates_dbpedia.owl'
                     ):
    """Generate the T-Box vector database (databases containing all the
    classes and predicates from the ontologies specified in the
    SemanticStore)
    """
    # Load the classes and predicates into vector stores

    print('Loading classes...')
    classes = tbox.get_classes_embedding_strings()
    print(f'Number of classes: {len(classes)}')
    with open(debug_classes_file, 'w') as f:
        for c in classes:
            f.write(c + '\n')

    print('Loading predicates...')
    predicates = tbox.get_predicates_embedding_strings()
    print(f'Number of predicates: {len(predicates)}')
    with open(debug_predicates_file, 'w') as f:
        for p in predicates:
            f.write(p + '\n')

    # Storage into vector databases
    print('Storing classes...')
    tbox.store_classes(classes)

    print('Storing predicates...')
    tbox.store_predicates(predicates)


if __name__ == '__main__':
    from langchain_community.llms import FakeListLLM
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings

    print('Initializing embeddings...')
    # Embedding function to be used by vector stores
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        show_progress_bar=True
    )

    print('Initializing T-Box...')
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

    print('Generating T-Box databases...')
    # Generate the classes and predicates vector databases
    generate_tbox_db(tbox)
