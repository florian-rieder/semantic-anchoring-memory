"""
Purpose: get the predicate or class which most closely resembles our input
"""

from typing import List
from tqdm import tqdm
import shutil
import re

from rdflib import BNode, Namespace, Graph, URIRef, Literal
from langchain_core.vectorstores import VectorStore


class TBoxLoader():
    def __init__(self, ontologies_paths: str):
        self.graph = Graph()
        for path in ontologies_paths:
            self.graph.parse(path)

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

        Improvement idea 2:
        Instead of storing RDF inside the vector store, store the data
        relevant to a similarity search only, one item per line.
        """

        query = """
        SELECT DISTINCT ?property ?propertyType ?domain ?range ?comment ?label
        WHERE {
            ?property rdf:type ?propertyType .
            VALUES ?propertyType { owl:ObjectProperty owl:DatatypeProperty }

            OPTIONAL { ?property rdfs:label ?label FILTER(LANG(?label) = 'en'). }
            OPTIONAL { ?property rdfs:comment ?comment FILTER(LANG(?comment) = 'en'). }
            OPTIONAL { ?property rdfs:domain ?domain . }
            OPTIONAL { ?property rdfs:range ?range . }
        }
        """

        result = self.graph.query(query)

        # Build RDF/XML string for each predicate
        predicates_rdf = []
        for row in tqdm(result):
            #predicate_storage_lines = []
            predicate_uri = str(row['property']) if row['property'] else None
            property_type = str(row['propertyType']) if row['propertyType'] else None
            label = str(row['label']) if row['label'] else None
            comment = str(row['comment']) if row['comment'] else None
            domain = str(row['domain']) if row['domain'] else None
            range_ = str(row['range']) if row['range'] else None


            predicate_rdf = '\n'.join(p for p in (
                f'<rdf:Description rdf:about="{predicate_uri}">',
                f'    <rdf:type rdf:resource="{property_type}"/>',
                f'    <rdfs:label>{label}</rdfs:label>' if label else '',
                f'    <rdfs:comment>{comment}</rdfs:comment>' if comment else '',
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
        """

        query = """
        SELECT DISTINCT ?class ?label ?comment ?subClassOf
        WHERE {
            ?class rdf:type owl:Class .
            OPTIONAL { ?class rdfs:label ?label FILTER(LANG(?label) = 'en'). }
            OPTIONAL { ?class rdfs:comment ?comment FILTER(LANG(?comment) = 'en'). }
            OPTIONAL { ?class rdfs:subClassOf ?subClassOf . }
        }
        """

        result = self.graph.query(query)

        # Parse results
        classes_rdf = {}
        for row in tqdm(result):
            class_uri = str(row['class'])
            label = str(row['label']) if row['label'] else None
            comment = str(row['comment']) if row['comment'] else None
            subClassOf = str(row['subClassOf']) if row['subClassOf'] else None

            if class_uri not in classes_rdf:
                classes_rdf[class_uri] = {'label': label, 'comment': comment, 'subClassOf': []}

            if subClassOf:
                classes_rdf[class_uri]['subClassOf'].append(subClassOf)

        # Generate RDF/XML
        rdf_descriptions = []
        for class_uri, class_data in tqdm(classes_rdf.items()):
            label = class_data['label']
            comment = class_data['comment']
            subClassOf_list = class_data['subClassOf']

            # precompute optional fstrings
            subClassOf_rdf_lines = [f'    <rdfs:subClassOf rdf:resource="{sc}"/>' for sc in subClassOf_list if sc]
            subClassOf_rdf = "\n".join(subClassOf_rdf_lines)
            label = f'    <rdfs:label>{label}</rdfs:label>\n' if label else ''
            comment = f'    <rdfs:comment>{comment}</rdfs:comment>\n' if comment else ''

            class_rdf_description = (
                f'<rdf:Description rdf:about="{class_uri}">\n'
                '    <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#Class"/>\n'
                f'{label}'
                f'{comment}'
                f'{subClassOf_rdf}\n'
                '</rdf:Description>'
            )

            rdf_descriptions.append(class_rdf_description)

        return set(rdf_descriptions)


class TBoxStorage():
    def __init__(self, predicates_db: VectorStore, classes_db: VectorStore):
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
    
    def encode_triplets(self, triplets : "list[tuple[str, str, str]]"):
        
        for triplet in triplets:
            subject, predicate, object = triplet
            if triplet[0] == 'User':
                subject_class = URIRef('http://dbpedia.org/ontology/Person')
            
            type_, value = [s.strip() for s in object.split(':')]
            if type_ == 'D':
                object = Literal(value)
            elif type_ == 'O':
                pass

            predicate = self.query_predicates(predicate)

    @staticmethod
    def _split_list(input_list, chunk_size):
        "Split a list into chunks of the given chunk size"
        # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]


def generate_tbox_db(ontologies_paths: "list[str]", store: TBoxStorage):
    # Load the classes and predicates into vector stores

    print('Loading T-Box...')
    loader = TBoxLoader(ontologies_paths)

    print('Loading classes...')
    classes = loader.load_classes()
    print(f'Number of classes: {len(classes)}')
    with open('ontologies/classes.owl', 'w') as f:
        for c in classes:
            f.write(c + '\n')

    print('Loading predicates...')
    predicates = loader.load_predicates()
    print(f'Number of predicates: {len(predicates)}')
    with open('ontologies/predicates.owl', 'w') as f:
        for p in predicates:
            f.write(p + '\n')

    # Storage into vector databases
    if input('Delete vector db ? (y/n) ').lower() == 'y':
        print('Deleting vector db...')
        shutil.rmtree('.database/vector_db/oa_predicates_db')
        shutil.rmtree('.database/vector_db/oa_classes_db')

    print('Storing predicates...')
    store.store_predicates(predicates)

    print('Storing classes...')
    store.store_classes(classes)


if __name__ == '__main__':
    from langchain.vectorstores import Chroma

    print('Initializing embeddings...')

    from langchain.embeddings.openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        show_progress_bar=True
    )

    # Different possible embeddings:

    # from langchain.embeddings import HuggingFaceEmbeddings
    # # simple and cheap option
    # embeddings = HuggingFaceEmbeddings()

    # embeddings = HuggingFaceEmbeddings(
    #     model_name="thenlper/gte-large",
    #     model_kwargs={"device": "cuda"},
    #     encode_kwargs={"normalize_embeddings": True},
    # )

    print('Initializing vector stores...')
    store = TBoxStorage(
        predicates_db=Chroma(
            persist_directory='./database/vector_db/oa_predicates_db',
            embedding_function=embeddings
        ),
        classes_db=Chroma(
            persist_directory='./database/vector_db/oa_classes_db',
            embedding_function=embeddings
        )
    )

    ontologies_paths = [
        'ontologies/dbpedia.owl',  # general ontology
        'http://xmlns.com/foaf/spec/index.rdf'  # people ontology
    ]

    generate_tbox_db(ontologies_paths, store)

    # Testing
    print('Test predicates:')
    test_predicates = [
        'has sister', 'has friend', 'likes', 'is good at',
        'works on', 'has', 'lives in', 'owns', 'has birthday'
    ]
    for p in test_predicates:
        print(f'{p}: {store.query_predicates(p)[0]}\n')

    print('Test classes')
    test_classes = [
        'person', 'place', 'city', 'concept',
        'US Army', 'friend', 'brother'
    ]
    for c in test_classes:
        print(f'{c}: {store.query_classes(c)[0]}\n')
