import os

from rdflib import Graph, URIRef

from langchain_core.vectorstores import VectorStore


class ABox():
    def __init__(self,
                 entities_store: VectorStore,
                 memory_base_path: str = 'ontologies/base_knowledge.ttl',
                 memory_path: str = 'database/_memories/knowledge.ttl',
                 memory_format: str = 'turtle'
                 ):
        self.graph: Graph = Graph()
        self.memory_base_path: str = memory_base_path
        self.memory_path: str = memory_path
        self.entities_db: VectorStore = entities_store
        self.memory_format: str = memory_format

        # Open the memory graph on load
        if os.path.exists(self.memory_path):
            self.graph.parse(self.memory_path)
        else:
            # first time setting up the memory A-Box:
            # set up the memory with a predefined base
            self.graph.parse(self.memory_base_path)
            self.save_graph()

    def store_entities(self, entities: list[str]):
        self.entities_db.add_texts(entities)
        self.entities_db.persist()

    def query_entities(self, query: str, k=4):
        """Get the k most similar entities from the entities db"""
        self.entities_db.similarity_search(query)
        return [d.page_content for d in self.entities_db.similarity_search(query, k)]

    def query_entities_with_score(self, query: str, threshold: float = 0.75, k=4):
        """Get the entities which are most relevant to the query, above
        the threshold"""
        # Query with score:
        results = self.entities_db.similarity_search_with_relevance_scores(
            query, k)
        # Get only the results with a relevance greater than the threshold
        matches = [d[0].page_content for d in results if d[1] > threshold]
        matches = [URIRef(m) for m in matches]
        return matches

    def save_graph(self):
        """Save the ABox graph to file"""
        self.graph.serialize(
            destination=self.memory_path,
            format=self.memory_format
        )
