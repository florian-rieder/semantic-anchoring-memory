from langchain_core.language_models import BaseLanguageModel

from langchain.chains import LLMChain

from rdflib import Namespace, URIRef, Literal, RDF, RDFS
import traceback
from urllib.parse import quote

from memory.semantic.prompts import CHOOSE_CLASS_PROMPT, CHOOSE_PREDICATE_PROMPT
from memory.semantic.abox import ABox, encode_entity_uri, decode_entity_uri
from memory.semantic.tbox import TBox

EX = Namespace("http://example.com/")

LITERAL_TYPES = (
    'http://www.w3.org/2001/XMLSchema#string',
    'http://www.w3.org/2000/01/rdf-schema#Literal',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString',
    'Literal'
)


class SemanticStore():
    def __init__(self,
                 tbox: TBox,
                 abox: ABox,
                 encoder_llm: BaseLanguageModel,
                 k_similar_classes: int = 8,
                 k_similar_predicates: int = 8,
                 entity_similarity_threshold: float = 0.75
                 ):
        self.tbox: TBox = tbox
        self.abox: ABox = abox
        self.encoder_llm: BaseLanguageModel = encoder_llm
        # How many similar classes to retrieve when trying to cast a class
        self.k_similar_classes = k_similar_classes
        # How many similar predicates to retrieve when trying to cast a predicate
        self.k_similar_predicates = k_similar_predicates
        # Normalized cosine similarity score over which the input similar enough
        # to the output that we'll consider they represent the same entity.
        self.entity_similarity_threshold = entity_similarity_threshold

    def encode_triplets(self, triplets: list[tuple[str, str, str]]):
        """Encode all given triplets into RDF"""
        encoded_triplets = [self._encode_triplet(t) for t in triplets]
        return encoded_triplets

    def _encode_triplet(self,
                        triplet: tuple[str, str, str],
                        ) -> dict:
        """Encode a triplet in natural language (ex. (User, is named, Bob))
        into a dictionary that lists the RDF classes and predicate that
        best represent the triple.
        """
        # triplet[0] and triplet[2] -> Cast to class
        # triplet[1] -> Cast to predicate
        # 1st attempt: take the whole triplet, and find out which classes and predicates are chosen

        # Subject is always an entity
        # Object can be a DataProperty or an ObjectProperty

        subject_class = self.encode_class(triplet, role='subject')
        object_class = self.encode_class(triplet, role='object')

        # Cases where the object is a literal
        if object_class in LITERAL_TYPES:
            object_class = 'Literal'

        predicate = self.encode_predicate(triplet, subject_class, object_class)

        encoded_triplet = {
            'subject': {
                'type': subject_class,
                'value': triplet[0]
            },
            'predicate': {
                'type': predicate
            },
            'object': {
                'type': object_class,
                'value': triplet[2]
            }
        }

        return encoded_triplet

    def memorize_encoded_triplets(self, encoded_triplets: list[dict]):
        """Takes the output from the encode_triplets() method, and add
        them to the knowledge graph, and save the knowledge graph."""

        for triplet in encoded_triplets:
            try:
                self._memorize_encoded_triplet(triplet)
            except Exception:
                # If an encoding fails, print the exception but continue
                # with the rest anyway
                traceback.print_exc()

        self.abox.save_graph()

    def _memorize_encoded_triplet(self, encoded_triplet: dict):
        """Takes in a dictionary from the encode_triplet method, and
        saves it to the knowledge memory graph. If an entity is not
        already in the memory graph, create a new node
        """

        # Convert string values to URIRef objects
        predicate_uri = encoded_triplet['predicate']['type']

        # SUBJECT
        subject = encoded_triplet['subject']['value']
        subject_uri = encode_entity_uri(subject)

        # Check if there is a sufficiently similar entity in entities db, and
        # therefore in the graph.
        existing_subject = self.resolve_memorized_entity(subject)

        print(f'Is subject already in memory ? {existing_subject}')

        # If there isn't such an entity, we consider this is a new entity, and
        # add it to the graph and entities db.
        if not existing_subject:
            subject_node = EX[subject_uri]
            print(f'Create new entity in memory: {subject_node}')
            # Add the new node to the graph
            self.abox.graph.add(
                (subject_node, RDF.type, encoded_triplet['subject']['type']))
            
            # Add the entity string as a label
            self.abox.graph.add(
                (subject_node, RDFS.label, Literal(subject)))
            # Store the new entity in the entities database
            self.abox.store_entities([subject])
        else:
            subject_node = EX[existing_subject]

        # OBJECT
        object_ = encoded_triplet['object']['value']
        object_uri = encode_entity_uri(object_)
        object_type = encoded_triplet['object']['type']
        object_node = None

        # Cases where the object is a Literal
        if str(object_type) == 'Literal':
            object_node = Literal(object_)

        # If the predicate is RDF.type, the object is not an entity but an
        # abstract category (like Artist, or Scientist)
        elif predicate_uri == RDF.type:
            object_node = object_type

        # Otherwise, the object is an ObjectProperty
        else:
            # First look in the entities DB to see if we already have
            # such an entity in memory
            existing_object = self.resolve_memorized_entity(object_)

            print(f'Is object already in memory ? {object_node}')

            # If the entity doesn't already exist, create a new one
            if not existing_object:
                object_node = EX[object_uri]
                print(f'Create new entity in memory: {object_node}')

                # Add the new node to the graph
                self.abox.graph.add(
                    (object_node, RDF.type, encoded_triplet['object']['type']))

                self.abox.graph.add(
                    (object_node, RDFS.label, Literal(object_)))
                # Store the new entity in the entities database
                self.abox.store_entities([object_])
            else:
                object_node = EX[existing_object]

        # PREDICATE
        # Add the triple to the Graph
        print((str(subject_node), str(predicate_uri), str(object_node)))
        self.abox.graph.add((subject_node, predicate_uri, object_node))

    def encode_class(self, triplet: tuple[str, str, str], role: str) -> URIRef:
        """Use an LLM to choose the best class to represent the subject
        or object of a triple"""

        triplet_str = str(triplet)
        subject = triplet[0]
        object_ = triplet[2]

        # Define class query and choice intent
        if role == 'object':
            query = f'{triplet_str}: RDF for object "{object_}"'
            intent = f'Get the correct class for object "{object_}" contained in the triplet {triplet_str}'
        elif role == 'subject':
            query = f'{triplet_str}: RDF for subject "{subject}"'
            intent = f'Get the correct class for subject "{subject}" contained in the triplet {triplet_str}'
        else:
            raise ValueError(
                f'Unexpected role {role}. Must be either "subject" or "object".')

        # Query the Classes DB for similar classes
        entity_classes = self.tbox.query_classes(
            query, k=self.k_similar_classes)

        # Get the URIs of the similar classes
        class_properties = self.tbox._get_properties_from_embedding_strings(
            entity_classes
        )

        possible_classes = list()
        # Get all the parent classes
        for uriref, _ in class_properties.items():
            possible_classes.append(uriref)
            parents = self.tbox.get_all_parent_classes(uriref)
            possible_classes += parents

        # The object of a triple can be a literal value
        if role == 'object':
            possible_classes.append('Literal')

        # Remove duplicates
        possible_classes = list(set(possible_classes))

        # Use LLM to choose the best class
        chosen_class = choose_class(
            intent=intent,
            classes=possible_classes,
            llm=self.encoder_llm
        )

        # If the LLM added stuff after the URI, cut off anything after the
        # first space and hope for the best
        chosen_class = chosen_class.strip().split(' ')[0].strip()

        # Print important info used by the LLM to reason
        print('Encoding class')
        print(query)
        print('Possible classes:')
        print('\n'.join(possible_classes))
        print('Chosen class:')
        print(chosen_class)
        print()

        return URIRef(chosen_class)

    def encode_predicate(self,
                         triplet: tuple[str, str, str],
                         subject_class: URIRef,
                         object_class: URIRef
                         ) -> URIRef:
        """Returns the best predicate URI to represent the predicate in
        the given triple"""

        # Build the query for the predicates vector database
        predicate_query = f'{str(triplet)}: RDF for predicate representing "{triplet[1]}"'

        # Get k possibly relevant predicates
        results = self.tbox.query_predicates(
            predicate_query, k=self.k_similar_predicates)

        # Get the domain and range of each predicate
        predicates_properties = self.tbox._get_properties_from_embedding_strings(
            results,
            {
                'domain': RDFS.domain,
                'range': RDFS.range,
                # 'type': RDF.type
            }
        )

        # Format the strings detailing the predicate's URI, its range and
        # domain for each predicate
        possible_predicates = list()
        for uriref, properties in predicates_properties.items():
            # list to string the domain and range
            domain = ", ".join(str(p) for p in properties['domain']) if len(
                properties['domain']) > 0 else 'Any'
            range_ = ", ".join(str(p) for p in properties['range']) if len(
                properties['range']) > 0 else 'Any'

            # Format the string describing this predicate
            pred_str = f'{str(uriref)} (domain: {domain}; range: {range_})'
            possible_predicates.append(pred_str)

        # Shortcut for appartenance to categories
        # 'is' is a common relationship that can be described by rdf:type
        if 'is' in triplet[1].split(' '):
            possible_predicates.append(RDF.type)

        # Use an LLM to choose the best predicate to use to represent the
        # relationship between the subject and object.
        intent = f'Get the correct predicate for "{triplet[1]}" contained in triplet {str(triplet)}. The subject class is {str(subject_class)}, and the object class is {str(object_class)}.'
        chosen_predicate = choose_predicate(
            intent=intent,
            predicates=possible_predicates,
            llm=self.encoder_llm
        )

        # If the LLM added stuff after the URI, cut off anything after the
        # first space and hope for the best
        chosen_predicate = chosen_predicate.strip().split(' ')[0].strip()

        # Print important info used by the LLM to reason
        print('Encoding predicate:')
        print(predicate_query)
        print('Possible predicates:')
        print('\n'.join(possible_predicates))
        print('Chosen predicate:')
        print(chosen_predicate)
        print()

        return URIRef(chosen_predicate)

    def resolve_memorized_entity(self, new_entity) -> URIRef:
        """Searches the memory to see if the new entity is already in memory.
        Returns the URIRef of the memorized entity, or None if no entity is found
        """
        # Verify if the object is an entity that's already in memory
        subject_entity_query = f"{new_entity}"
        similar_objects_in_memory = self.abox.query_sufficiently_similar_entity(
            subject_entity_query, threshold=self.entity_similarity_threshold)

        # If there is exactly one match, then we're probably talking about the
        # same entity
        if len(similar_objects_in_memory) == 1:
            return URIRef(encode_entity_uri(similar_objects_in_memory[0]))
        elif len(similar_objects_in_memory) > 1:
            # TODO: Ask the user to clarify ?
            # Maybe store incoherences in an "interrogations" db, which the
            # chatbot will try to satisfy by asking the user to clarify

            # But for now do exactly the same
            return URIRef(encode_entity_uri(similar_objects_in_memory[0]))
        else:
            return None


def choose_predicate(intent: str, predicates: list[str], llm) -> str:
    """Use an LLM to choose the predicate from a list of predicates, which is the
    most relevant to the intent"""
    chain = LLMChain(
        llm=llm,
        prompt=CHOOSE_PREDICATE_PROMPT,
    )

    chosen_predicate = chain.predict(
        predicates="\n".join([str(p) for p in predicates]),
        intent=intent,
    )

    return chosen_predicate


def choose_class(intent: str, classes: list[str], llm) -> str:
    """Use an LLM to choose a class from a list of classes, which is the
    most relevant to the intent"""
    chain = LLMChain(
        llm=llm,
        prompt=CHOOSE_CLASS_PROMPT,
    )

    chosen_class = chain.predict(
        classes="\n".join([str(c) for c in classes]),
        intent=intent,
    )

    return chosen_class
