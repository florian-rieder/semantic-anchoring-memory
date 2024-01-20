from langchain_core.vectorstores import VectorStore
from langchain_core.language_models import BaseLanguageModel

from langchain.chains import LLMChain

from rdflib import Namespace, URIRef, Literal, RDF, RDFS

from server.memory.prompts import CHOOSE_CLASS_PROMPT, CHOOSE_PREDICATE_PROMPT
from server.memory.semantic.abox import ABox
from server.memory.semantic.tbox import TBox

EX = Namespace("http://example.com/")

LITERAL_TYPES = (
    'http://www.w3.org/2001/XMLSchema#string',
    'http://www.w3.org/2000/01/rdf-schema#Literal',
    'http://www.w3.org/1999/02/22-rdf-syntax-ns#langString',
    'Literal'
)


class TBoxStorage():
    def __init__(self,
                 predicates_db: VectorStore,
                 classes_db: VectorStore,
                 tbox: TBox,
                 abox: ABox,
                 encoder_llm: BaseLanguageModel
                 ):
        self.pred_db: VectorStore = predicates_db
        self.class_db: VectorStore = classes_db
        self.tbox: TBox = tbox
        self.abox: ABox = abox
        self.encoder_llm: BaseLanguageModel = encoder_llm

    def store_predicates(self,
                         predicates_embedding_strings: list[str]
                         ) -> None:
        self.pred_db.add_texts(predicates_embedding_strings)
        self.pred_db.persist()

    def store_classes(self,
                      classes_embedding_strings: list[str]
                      ) -> None:
        self.class_db.add_texts(classes_embedding_strings)
        self.class_db.persist()

    def query_predicates(self,
                         query: str,
                         k: int = 4
                         ) -> str:
        """
        Returns k predicates which are most similar to the input query.
        """
        return [d.page_content for d in self.pred_db.similarity_search(query, k=k)]

    def query_classes(self,
                      query: str,
                      k: int = 4
                      ) -> str:
        """
        Returns k predicates which are most similar to the input query.
        """
        return [d.page_content for d in self.class_db.similarity_search(query, k)]

    def encode_triplet(self,
                       triplet: tuple[str, str, str],
                       ) -> tuple[str, str, str]:
        """Encode a triplet in natural language (ex. (User, is named, Bob))
        into a dictionary that lists the RDF classes and predicate that
        best represent the triple.
        """
        # triplet[0] and triplet[2] -> Cast to class
        # triplet[1] -> Cast to predicate
        # 1st attempt: take the whole triplet, and find out which classes and predicates are chosen

        # subject is always an entity
        # subject_entities = abox.get_entities(triplet[0])
        # object can be a DataProperty or an ObjectProperty
        # object_entities = abox.get_entities(triplet[2])

        subject_class = self.encode_class(triplet, role='subject')
        object_class = self.encode_class(triplet, role='object')

        # Cases where the object is a literal
        if object_class in LITERAL_TYPES:
            object_class = RDF.Literal

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

    def memorize_encoded_triplet(self, encoded_triplet: dict):
        """ Takes in a dictionary from the encode_triplet method, and
        saves it to the knowledge memory graph
        """

        # Step 3: Convert string values to URIRef objects
        subject_uri = EX[encoded_triplet['subject']['value']]
        predicate_uri = encoded_triplet['predicate']['type']
        object_uri = EX[encoded_triplet['object']['value']]

        # TODO: Check if there are similar entities in the graph ?
        # Should we put every entity in the graph into a vector db
        # for similarity search ?

        # Cases where the object is a literal
        if encoded_triplet['object']['type'] == RDF.Literal:
            object_node = Literal(encoded_triplet['object']['value'])
        else:
            object_node = EX[encoded_triplet['object']['value']]

        # Step 4: Add Triples to the Graph
        self.abox.add(
            (subject_uri, RDF.type, encoded_triplet['subject']['type']))
        self.abox.add((subject_uri, predicate_uri, object_uri))
        self.abox.add(
            (subject_uri, RDF.type, encoded_triplet['subject']['type']))
        self.abox.add((subject_uri, predicate_uri, object_node))

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

        entity_classes = self.query_classes(query)
        class_properties = self.tbox._get_properties_from_embedding_strings(
            # DEBUG: remove [0] to take all results into account
            entity_classes
        )
        print(class_properties)

        possible_classes = list()
        # get all the parent classes
        for uriref, _ in class_properties.items():
            possible_classes.append(uriref)
            parents = self.tbox.get_all_parent_classes(uriref)
            possible_classes += parents

        # remove duplicates
        possible_classes = list(set(possible_classes))

        # Use LLM to choose the best class
        chosen_class = choose_class(
            intent=intent,
            classes=possible_classes,
            llm=self.encoder_llm
        )

        print('Chosen class:')
        print(chosen_class)

        return URIRef(chosen_class)

    def encode_predicate(self,
                         triplet: tuple[str, str, str],
                         subject_class: URIRef,
                         object_class: URIRef,
                         num_predicates_to_get: int = 8
                         ) -> URIRef:
        # Build the query for the predicates vector database
        predicate_query = f'{str(triplet)}: RDF for predicate representing "{triplet[1]}"'
        
        # Get k possibly relevant predicates
        results = self.query_predicates(
            predicate_query, k=num_predicates_to_get)
        
        # Get the domain and range of each predicate
        predicates_properties = self.tbox._get_properties_from_embedding_strings(
            results,
            {
                'domain': RDFS.domain,
                'range': RDFS.range,
                #'type': RDF.type
            }
        )

        # Format the strings detailing the predicate's URI, its range and
        # domain for each predicate
        possible_predicates = list()
        for uriref, properties in predicates_properties.items():
            # list to string
            domain = ", ".join(str(p) for p in properties['domain']) if len(
                properties['domain']) > 0 else 'Any'
            range_ = ", ".join(str(p) for p in properties['range']) if len(
                properties['range']) > 0 else 'Any'

            pred_str = f'{str(uriref)} (domain: {domain}; range: {range_})'
            print(pred_str)
            possible_predicates.append(pred_str)

        # Use an LLM to choose the best predicate to use to represent the
        # relationship between the subject and object.
        intent = f'Get the correct predicate for "{triplet[1]}" contained in triplet {str(triplet)}. The subject class is {str(subject_class)}, and the object class is {str(object_class)}.'
        chosen_predicate = choose_predicate(
            intent=intent,
            predicates=possible_predicates,
            llm=self.encoder_llm
        )

        print('Chosen predicate:')
        print(chosen_predicate)
        return URIRef(chosen_predicate)


def generate_tbox_db(store: TBoxStorage):
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


def choose_predicate(intent: str, predicates: list[str], llm) -> str:
    chain = LLMChain(
        llm=llm,
        prompt=CHOOSE_PREDICATE_PROMPT,
        verbose=True
    )

    chosen_predicate = chain.predict(
        predicates="\n".join([str(p) for p in predicates]),
        intent=intent
    )

    return chosen_predicate


def choose_class(intent: str, classes: list[str], llm) -> str:
    chain = LLMChain(
        llm=llm,
        prompt=CHOOSE_CLASS_PROMPT,
        verbose=True
    )

    chosen_class = chain.predict(
        classes="\n".join([str(c) for c in classes]),
        intent=intent,
    )

    return chosen_class
