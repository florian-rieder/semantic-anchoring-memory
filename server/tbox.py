"""
Purpose: get the predicate or class which most closely resembles our input
"""

from typing import List
from tqdm import tqdm
import shutil
import re
import queue


from rdflib import BNode, Namespace, Graph, URIRef, Literal, RDF, RDFS
from langchain_core.vectorstores import VectorStore

import pprint

class TBoxLoader():
    def __init__(self, ontologies_paths: list[str]):
        self.graph = Graph()
        for path in ontologies_paths:
            self.graph.parse(path)

    def load_predicates(self) -> list[str]:
        """
        Summary
        -------
        Store all of the predicates in the given ontology inside a
        vector store.

        Returns
        -------
        list[str]:
            A list of predicate embedding strings, which are strings
            containing extra information about the classes described in
            the given ontology, to be stored in a vector database.

        Notes
        -----
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

            OPTIONAL { ?property rdfs:label ?label FILTER(LANG(?label) = 'en' || LANG(?label) = ''). }
            OPTIONAL { ?property rdfs:comment ?comment FILTER(LANG(?comment) = 'en' || LANG(?comment) = ''). }
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

    def load_classes(self) -> list[str]:
        """
        Summary
        -------
        Store all of the classes in the given ontology inside a
        vector store.

        Returns
        -------
        list[str]:
            A list of class embedding strings, which are strings
            containing extra information about the classes described in
            the given ontology, to be stored in a vector database.
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
    
    def get_properties(self, subject: URIRef, properties_to_get: dict) -> dict[str, URIRef]:
        properties = dict()
        for property_name, property in properties_to_get.items():
            triples = list(self.graph.triples((subject, property, None)))
            property_values = [t[2] for t in triples]
            properties[property_name] = property_values
        return properties
    
    def _get_properties_from_embedding_strings(self, embedding_strings: list[str], properties_to_get: dict[str, URIRef]) -> dict[str, dict[str, URIRef]]:
        subject_properties = {}
        for idx, pred_str in enumerate(embedding_strings):

            # Grab the URI
            pattern = re.compile(r'<rdf:Description rdf:about="((?:.)+?)">')
            uri = pattern.match(pred_str).group(1)

            properties = self.tbox.get_properties(
                URIRef(uri),
                properties_to_get
            )
            properties['priority'] = idx
            subject_properties[uri] = properties
        return subject_properties

    def get_parent_classes(self, target_class: URIRef) -> list[URIRef]:
        """Get all direct parents of the given class"""
        return [t[2] for t in self.graph.triples((target_class, RDFS.subClassOf, None))]

    def get_all_parent_classes(self, target_class: URIRef) -> list[URIRef]:
        """Get all the parent classes in the hierarchy"""
        
        parents = self.get_parent_classes(target_class)

        nodes_to_check = queue.Queue()
        for parent in parents:
            nodes_to_check.put(parent)

        while not nodes_to_check.empty():
            next_parents = self.get_parent_classes(nodes_to_check.get())
            for parent in next_parents:
                nodes_to_check.put(parent)
                parents.append(parent)

        return list(set(parents))

    

class TBoxStorage():
    def __init__(self,
                 predicates_db: VectorStore,
                 classes_db: VectorStore,
                 loader: TBoxLoader
                ):
        self.pred_db : VectorStore = predicates_db
        self.class_db : VectorStore = classes_db
        self.tbox : TBoxLoader = loader

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
    
    def encode_triplet(self, triplet: tuple[str, str, str]) -> tuple[str, str, str]:
        # triplet[0] and triplet[2] -> Cast to class
        # triplet[1] -> Cast to predicate

        # 1st attempt: take the whole triplet, and find out which classes and predicates are chosen

        # subject_query = f'{str(triplet)}: RDF for subject "{triplet[0]}"'
        # subject = self.query_classes(subject_query)
        # object_properties = self.tbox._get_properties_from_embedding_strings(
        #     [object_[0]],
        #     {
        #         'parent_class': RDFS.subClassOf,
        #     }
        # )
        
        # object_query = f'{str(triplet)}: RDF for object "{triplet[2]}"'
        # object_ = self.query_classes(object_query)
        # object_properties = self.tbox._get_properties_from_embedding_strings(
        #     [object_[0]],
        #     {
        #         'parent_class': RDFS.subClassOf,
        #     }
        # )

        predicate_query = f'{str(triplet)}: RDF for predicate representing "{triplet[1]}"'
        results = self.query_predicates(predicate_query)
        predicates_properties = self.tbox._get_properties_from_embedding_strings(
            [results][0],
            {
                    'domain': RDFS.domain,
                    'range': RDFS.range,
                    'type': RDF.type
            }
        )

        possible_domains = []
        possible_ranges = []

        for predicate_properties in predicates_properties:
            domain_parents = self.tbox.get_all_parent_classes(predicate_properties['domain'])
            range_parents = self.tbox.get_all_parent_classes(predicate_properties['range'])

            possible_domains.append(domain_parents)
            possible_ranges.append(range_parents)
        
        encoded_triplet = ()



    @staticmethod
    def _split_list(input_list, chunk_size):
        "Split a list into chunks of the given chunk size"
        # https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
        for i in range(0, len(input_list), chunk_size):
            yield input_list[i:i + chunk_size]

def generate_tbox_db(store: TBoxStorage):
    # Load the classes and predicates into vector stores

    print('Loading classes...')
    classes = store.tbox.load_classes()
    print(f'Number of classes: {len(classes)}')
    with open('ontologies/classes.owl', 'w') as f:
        for c in classes:
            f.write(c + '\n')

    print('Loading predicates...')
    predicates = store.tbox.load_predicates()
    print(f'Number of predicates: {len(predicates)}')
    with open('ontologies/predicates.owl', 'w') as f:
        for p in predicates:
            f.write(p + '\n')

    # Storage into vector databases
    print('Storing classes...')
    store.store_classes(classes)

    print('Storing predicates...')
    store.store_predicates(predicates)


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

    ontologies_paths = [
        'ontologies/dbpedia.owl',  # general ontology
        'ontologies/foaf.owl'
        #'http://xmlns.com/foaf/spec/index.rdf'  # people ontology
    ]

    print('Initializing T-Box...')
    tbox_loader = TBoxLoader(ontologies_paths)

    # print('Initializing vector stores...')
    # store = TBoxStorage(
    #     predicates_db=Chroma(
    #         persist_directory='./database/vector_db/oa_predicates_db',
    #         embedding_function=embeddings
    #     ),
    #     classes_db=Chroma(
    #         persist_directory='./database/vector_db/oa_classes_db',
    #         embedding_function=embeddings
    #     ),
    #     loader=tbox_loader
    # )

    #generate_tbox_db(store)

    print('getting all parent classes...')
    print(tbox_loader.get_all_parent_classes(URIRef('http://dbpedia.org/ontology/StormSurge')))
    #store.encode_triplet(('User', 'is named', 'Florian'))

    # Testing

    #results = ["""<rdf:Description rdf:about="http://dbpedia.org/ontology/peopleName">
    #     <rdf:type rdf:resource="http://www.w3.org/2002/07/owl#DatatypeProperty"/>
    #     <rdfs:label>peopleName</rdfs:label>
    #     <rdfs:comment>Name for the people inhabiting a place, eg Ankara->Ankariotes, Bulgaria->Bulgarians</rdfs:comment>
    #     <rdfs:domain rdf:resource="http://dbpedia.org/ontology/PopulatedPlace"/>
    #     <rdfs:range rdf:resource="http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"/>
    # </rdf:Description>
    # """]
    # store.tbox._get_properties_from_embedding_strings(results)
    # print('Test predicates:')
    # # test_predicates = [
    # #     'has sister', 'has friend', 'likes', 'is good at',
    # #     'works on', 'has', 'lives in', 'owns', 'has birthday',
    # #     'is named'
    # # ]
    # test_predicates = [
    #     '(User, has sister, Shana): for predicate "has sister"'
    # ]
    # for p in test_predicates:
    #     print(f'{p}: {store.query_predicates(p)[0]}\n')

    # print('Test classes')
    # test_classes = [
    #     'person', 'place', 'city', 'concept',
    #     'US Army', 'friend', 'brother'
    # ]
    # for c in test_classes:
    #     print(f'{c}: {store.query_classes(c)[0]}\n')
