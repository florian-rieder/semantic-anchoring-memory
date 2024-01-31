import os

from rdflib import Graph, URIRef
from urllib.parse import unquote

from langchain_core.vectorstores import VectorStore


class ABox():
    def __init__(self,
                 entities_store: VectorStore,
                 memory_base_path: str,
                 memory_path: str,
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
            # If no base knowledge is defined, simply don't create base
            # knowledge
            if not self.memory_base_path:
                return

            # first time setting up the memory A-Box:
            # set up the memory with a predefined base
            self.graph.parse(self.memory_base_path)

            # Add all entities present in the base knowledge to the entities
            # database !
            entities_in_base_knowledge = list(set(self.graph.subjects()))
            self.store_entities(entities_in_base_knowledge)

            self.save_graph()

    def store_entities(self, entities: list[str]):
        self.entities_db.add_texts(entities)
        self.entities_db.persist()

    def query_entities(self, query: str, k=4):
        """Get the k most similar entities from the entities db"""
        self.entities_db.similarity_search(query)
        return [d.page_content for d in self.entities_db.similarity_search(query, k)]

    def query_entities_with_score(self,
                                  query: str,
                                  threshold: float = 0.75,
                                  k=4
                                  ) -> list[str]:
        """Get the entities which are most relevant to the query, above
        the threshold"""
        # Query with score:
        results = self.entities_db.similarity_search_with_relevance_scores(
            query, k)
        # Get only the results with a relevance greater than the threshold
        matches = [d[0].page_content for d in results if d[1] > threshold]
        return matches

    def get_entity_knowledge(self, entity: str) -> list[str]:
        # Get similar entities using a similarity search in the entities database
        similar_entities = self.query_entities(entity)
        print('Similar entities')
        print(similar_entities)
        entity_node = URIRef(similar_entities[0])
        print('Chosen Entity:')
        print(entity_node)

        # Get all the knowledge about this entity
        # TODO: if more entities are revealed, gather knowledge about them also
        knowledge = list()
        for pred, obj in self.graph.predicate_objects(entity_node):
            # Get the last bit of the URI
            # ex. https://example.com/Bob -> Bob
            pred = unquote(str(pred)).split("/")[-1].split('#')[-1]
            obj = unquote(str(obj)).split("/")[-1].split('#')[-1]

            knowledge_bit = f"{pred} {obj}"
            knowledge.append(knowledge_bit)

        # TODO: Filter knowledge to only what is relevant to the conversation

        return knowledge

    def save_graph(self):
        """Save the ABox graph to file"""
        self.graph.serialize(
            destination=self.memory_path,
            format=self.memory_format
        )
