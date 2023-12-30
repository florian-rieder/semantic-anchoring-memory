from rdflib import Graph
from langchain_core.vectorstores import VectorStore

from typing import List


class TBoxLoader():
    def __init__(self, ontology_path: str):
        self.ontology_path = ontology_path

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
        graph = Graph().parse(self.ontology_path)

        query = """
        SELECT DISTINCT ?property
        WHERE {
        ?property rdf:type owl:ObjectProperty
        }
        """

        result = graph.query(query)

        # Extract specific properties from the query result
        predicates = set(str(row['property']) for row in result)
        print(f"Number of predicates: {len(predicates)}")
        return predicates

    def load_classes(self):
        graph = Graph().parse(self.ontology_path)

        query = """
        SELECT DISTINCT ?class
        WHERE {
        ?class rdf:type owl:Class.
        }
        """

        result = graph.query(query)
        classes = set(str(row['class']) for row in result)

        print(f"Number of classes: {len(classes)}")
        return classes


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
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.embeddings import HuggingFaceEmbeddings

    # embeddings = OpenAIEmbeddings(
            #     model='text-embedding-ada-002',
            #     #show_progress_bar=True,)
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="thenlper/gte-large",
    #     model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True},
    # )

    TBox_path = 'ontologies/dbpedia_T_Box.owl'
    embeddings = HuggingFaceEmbeddings()

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
    

    # Load the classes and predicates into vector stores
    # c = TBoxLoader(TBox_path)
    # classes = c.load_classes()
    # predicates = c.load_predicates()
    
    # store.store_predicates(predicates)
    # store.store_classes(classes)


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


