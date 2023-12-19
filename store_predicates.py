from rdflib import Graph
from langchain_core.vectorstores import VectorStore


class PredicateStorage():
    def __init__(self, db : VectorStore):
        self.db = db

    def store_predicates(self, ontology_path: str):
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
        graph = Graph().parse(ontology_path)

        query = """
        SELECT DISTINCT ?property
        WHERE {
        ?property rdf:type owl:ObjectProperty
        }
        """

        result = graph.query(query)

        # Extract specific properties from the query result
        predicates = set(row['property'] for row in result)
        print(f"Number of predicates: {len(predicates)}")

        self.db.add_texts(predicates)
        self.db.persist()
    
        # split_docs_chunked = self._split_list(predicates, 1000)
        # for split_docs_chunk in split_docs_chunked:
        #     self.db.add_texts(split_docs_chunk)
        #     self.db.persist()


    def query_predicates(self, query: str) -> str:
        """
        Returns a single predicate which is most similar to the input query.
        """
        return [d.page_content for d in self.db.similarity_search(query)]

    @staticmethod
    def _split_list(input_list, chunk_size):
        "Split a list into chunks of the given chunk size"
        # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

if __name__ == '__main__':
    from langchain.vectorstores import Chroma
    from langchain.embeddings.openai import OpenAIEmbeddings

    ontology_path = "ontologies/dbpedia_T_box.owl"

    store = PredicateStorage(
        db=Chroma(
            persist_directory='./predicates_db',
            embedding_function=OpenAIEmbeddings(
                model='text-embedding-ada-002',
                #show_progress_bar=True,
            )
        )
    )

    #store.store_predicates(ontology_path)
    print(f"sister: {store.query_predicates('has sister')}")
    print(f"friend: {store.query_predicates('has friend')}")
    print(f"likes: {store.query_predicates('likes')}")
    print(f"is good at: {store.query_predicates('is good at')}")
    print(f"works on: {store.query_predicates('works on')}")
    print(f"has: {store.query_predicates('has')}")
    print(f"lives in: {store.query_predicates('lives in')}")
