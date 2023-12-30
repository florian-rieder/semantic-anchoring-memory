from rdflib import Graph
from langchain_core.vectorstores import VectorStore

from typing import List
from tqdm import tqdm


class TBoxLoader():
    def __init__(self, ontology_path: str):
        self.graph = Graph().parse(ontology_path)

    def load_predicates(self) -> List[str]:
        """
        Store all of the predicates in the given ontology inside a
        vector store.

        Improvement idea:
        Not only store the URI of the property, but a chunk of RDF XML
        (probably in the most compact format, something like turtle to
        save tokens) that contains all relevant information about the
        predicate (domain, range, comment (useful for the similarity
        search !))
        """

        query = """
        SELECT DISTINCT ?property ?domain ?range ?comment ?label
        WHERE {
            ?property rdf:type owl:ObjectProperty .
            OPTIONAL { ?property rdfs:label ?label FILTER(LANG(?label) = 'en'). }
            OPTIONAL { ?property rdfs:comment ?comment FILTER(LANG(?label) = 'en'). }
            OPTIONAL { ?property rdfs:domain ?domain . }
            OPTIONAL { ?property rdfs:range ?range . }
        }
        """

        result = self.graph.query(query)

        # Build RDF/XML string for each predicate
        predicates_rdf = []
        for row in tqdm(result):
            predicate_uri = str(row['property'])

            label = str(row['label'])
            comment = str(row['comment'])
            domain = str(row['domain'])
            range_ = str(row['range'])

            predicate_rdf = '\n'.join(p for p in (
            f'<rdf:Description rdf:about="{predicate_uri}">',
            '    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#ObjectProperty"/>',
            f'    <rdfs:label xml:lang="en">{label}</rdfs:label>' if label else '',
            f'    <rdfs:comment xml:lang="en">{comment}</rdfs:comment>' if comment else '',
            f'    <rdfs:domain rdf:resource="{domain}"/>' if domain else '',
            f'    <rdfs:range rdf:resource="{range_}"/>' if range_ else '',
            '</rdf:Description>'
            ) if p)

            predicates_rdf.append(predicate_rdf)

        return set(predicates_rdf)

    def load_classes(self) -> List[str]:
        """
        Store all of the classes in the given ontology inside a
        vector store.

        Improvement idea:
        Not only store the URI of the class, but a chunk of RDF XML
        (probably in the most compact format, something like turtle to
        save tokens) that contains all relevant information about the
        class (label, comment, subClassOf)
        """

        query = """
        SELECT DISTINCT ?class ?label ?comment ?subClassOf
        WHERE {
            ?class rdf:type owl:Class .
            OPTIONAL { ?class rdfs:label ?label FILTER(LANG(?label) = 'en'). }
            OPTIONAL { ?class rdfs:comment ?comment FILTER(LANG(?label) = 'en'). }
            OPTIONAL { ?class rdfs:subClassOf ?subClassOf . }
        }
        """

        result = self.graph.query(query)

        # Build RDF/XML string for each class
        classes_rdf = []
        for row in tqdm(result):
            class_uri = str(row['class'])

            label = str(row['label'])
            comment = str(row['comment'])
            subClassOf = str(row['subClassOf'])

            class_rdf = '\n'.join(p for p in (
                f'<rdf:Description rdf:about="{class_uri}">',
                '    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Class"/>',
                f'    <rdfs:label xml:lang="en">{label}</rdfs:label>' if label else '',
                f'    <rdfs:comment>{comment}</rdfs:comment>' if comment else '',
                f'    <rdfs:subClassOf rdf:resource="{subClassOf}"/>' if subClassOf else '',
                '</rdf:Description>'
            ) if p)

            classes_rdf.append(class_rdf)

        return set(classes_rdf)

class TBoxStorage():
    def __init__(self, predicates_db : VectorStore, classes_db : VectorStore):
        self.pred_db = predicates_db
        self.class_db = classes_db
    
    def store_predicates(self, predicates):
        self.pred_db.add_texts(predicates)
        self.pred_db.persist()
        # split_docs_chunked = self._split_list(predicates, 1000)
        # for split_docs_chunk in split_docs_chunked:
        #     self.db.add_texts(split_docs_chunk)
        #     self.db.persist()
    
    def store_classes(self, classes):
        self.class_db.add_texts(classes)
        self.class_db.persist()

    def query_predicates(self, query: str) -> str:
        """
        Returns a single predicate which is most similar to the input query.
        """
        return [d.page_content for d in self.pred_db.similarity_search(query)]
    
    def query_classes(self, query: str) -> str:
            """
            Returns a single predicate which is most similar to the input query.
            """
            return [d.page_content for d in self.class_db.similarity_search(query)]

    @staticmethod
    def _split_list(input_list, chunk_size):
        "Split a list into chunks of the given chunk size"
        # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]


if __name__ == '__main__':
    from langchain.vectorstores import Chroma
    from langchain.embeddings import HuggingFaceEmbeddings
    # from langchain.embeddings.openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings(
            #     model='text-embedding-ada-002',
            #     #show_progress_bar=True,)
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="thenlper/gte-large",
    #     model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True},
    # )

    print('Initializing embeddings...')
    embeddings = HuggingFaceEmbeddings()

    print('Initializing vector stores...')
    store = TBoxStorage(
        predicates_db=Chroma(
            persist_directory='./vector_db/hf_predicates_db',
            embedding_function = embeddings
        ),
        classes_db=Chroma(
            persist_directory='./vector_db/hf_classes_db',
                embedding_function=embeddings
        )
    )
    
    
    print('Loading T-Box...')
    TBox_path = 'ontologies/dbpedia_T_Box.owl'
    # Load the classes and predicates into vector stores
    loader = TBoxLoader(TBox_path)

    print('Loading classes...')
    classes = loader.load_classes()
    with open('classes.owl', 'w') as f:
        for c in classes:
            f.write(c)

    print('Loading predicates...')
    predicates = loader.load_predicates()
    with open('predicates.owl', 'w') as f:
        for p in predicates:
            f.write(p)
    
    print('Storing predicates...')
    store.store_predicates(predicates)

    print('Storing classes...')
    store.store_classes(classes)


    print('Test predictions:')
    print(f"sister: {store.query_predicates('has sister')}")
    print(f"friend: {store.query_predicates('has friend')}")
    print(f"likes: {store.query_predicates('likes')}")
    print(f"is good at: {store.query_predicates('is good at')}")
    print(f"works on: {store.query_predicates('works on')}")
    print(f"has: {store.query_predicates('has')}")
    print(f"lives in: {store.query_predicates('lives in')}")
    print('\n')
    print(f"person: {store.query_classes('person')}")
    print(f"place: {store.query_classes('place')}")
    print(f"concept: {store.query_classes('concept')}")
    print(f"US Army: {store.query_classes('US Army')}")


