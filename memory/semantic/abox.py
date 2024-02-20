import os

from rdflib import Graph, URIRef
from urllib.parse import quote, unquote
# from owlrl import DeductiveClosure, OWLRL_Semantics, interpret_owl_imports

from langchain_core.vectorstores import VectorStore

from config import COREFERENCE_SIMILARITY_THRESHOLD


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
        """
        Summary
        -------
        Store a list of entity strings in the entities db. An entity string is
        a string in natural language used as the descriptor of an entity
        in memory. For example "Yellow vests protests"

        Usage
        -----
        my_abox = Abox(...)
        abox.store_entities(["Yellow vests protests"])
        """
        self.entities_db.add_texts(entities)
        self.entities_db.persist()

    def query_entities(self, query: str, k=4):
        """Get the k most similar entities from the entities db"""
        self.entities_db.similarity_search(query)
        return [d.page_content for d in self.entities_db.similarity_search(query, k)]

    def query_sufficiently_similar_entity(self,
                                          query: str,
                                          threshold: float = COREFERENCE_SIMILARITY_THRESHOLD,
                                          k: int = 4
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
        entity_node = URIRef(encode_entity_uri(similar_entities[0]))
        print('Chosen Entity:')
        print(entity_node)

        # Get all the knowledge about this entity

        # First, let's use OWL-RL reasoner to infer additional properties,
        # semantically expanding the graph.
        # Get the vocabularies imported in the A-Box graph (dbpedia, foaf, etc)
        # interpret_owl_imports("AUTO", self.graph)
        # # Reason and expand the graph
        # DeductiveClosure(OWLRL_Semantics).expand(self.graph)

        # There is significant improvement opportunity in how we navigate the
        # graph here. Using reasoners and using various strategies for graph
        # navigation could allow for getting the most relevant information to
        # the conversation at hand.
        knowledge = list()
        for pred, obj in self.graph.predicate_objects(entity_node):
            # Transform relationship to a string akin to natural language, to
            # improve results when used in the prompt with the LLM.
            # Get the last bit of the URI: ex. https://example.com/Bob -> Bob
            pred = decode_entity_uri(str(pred)).split("/")[-1].split('#')[-1]
            obj = decode_entity_uri(str(obj)).split("/")[-1].split('#')[-1]

            # TODO: navigate the graph in order to retrieve the most
            # pertinent information

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


def encode_entity_uri(entity_string: str) -> str:
    return quote(entity_string.replace(' ', '_'))


def decode_entity_uri(entity_uri: str) -> str:
    return unquote(entity_uri).replace('_', ' ')
