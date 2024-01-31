"""
Purpose: get the predicate or class which most closely resembles our input
"""
import queue
import re
from tqdm import tqdm

from rdflib import Graph, URIRef, RDFS

from langchain_core.vectorstores import VectorStore


class TBox():
    def __init__(self,
                 ontologies_paths: list[str],
                 predicates_db: VectorStore,
                 classes_db: VectorStore,
                 ):
        self.graph = Graph()
        self.pred_db: VectorStore = predicates_db
        self.class_db: VectorStore = classes_db

        for path in ontologies_paths:
            self.graph.parse(path)

    def store_predicates(self,
                         predicates_embedding_strings: list[str]
                         ) -> None:
        """Add the list of predicate embedding strings to the predicates
        database"""
        self.pred_db.add_texts(predicates_embedding_strings)
        self.pred_db.persist()

    def store_classes(self,
                      classes_embedding_strings: list[str]
                      ) -> None:
        """Add the list of class embedding strings to the classes
        database"""
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


    def get_predicates_embedding_strings(self) -> list[str]:
        """
        Summary
        -------
        Get all of the predicates in the given ontology, transform them
        into embedding strings, which are strings containing extra
        information about the classes described in the given ontology,
        to be stored in a vector database.

        For the moment, those embedding strings are a bit of RDF/XML
        containing the predicate, its label, comment, domain and range.

        Returns
        -------
        list[str]:
            A list of predicate embedding strings.

        Notes
        -----
        Improvement idea:
        Not only store the URI of the property, but a chunk of RDF XML
        that contains all relevant information about the
        predicate (domain, range, comment (useful for the similarity
        search !))

        Improvement idea 2:
        - Instead of storing RDF inside the vector store, store the data
            relevant to a similarity search only, one item per line.
        - Use a more compact format, something like turtle to save tokens
            and tokens which don't add much to the similarity search.

        Add rdfs:subPropertyOf and rdfs:inverseOf?
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
            predicate_uri = str(row['property']) if row['property'] else None
            property_type = str(row['propertyType']
                                ) if row['propertyType'] else None
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

    def get_classes_embedding_strings(self) -> list[str]:
        """
        Summary
        -------
        Get all of the classes in the given ontology, transform them
        into embedding strings, which are strings containing extra
        information about the predicatess described in the given
        ontology, to be stored in a vector database.

        For the moment, those embedding strings are a bit of RDF/XML
        containing the class, its label, comment, and parent classes.

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
                classes_rdf[class_uri] = {
                    'label': label, 'comment': comment, 'subClassOf': []}

            if subClassOf:
                classes_rdf[class_uri]['subClassOf'].append(subClassOf)

        # Generate RDF/XML
        rdf_descriptions = []
        for class_uri, class_data in tqdm(classes_rdf.items()):
            label = class_data['label']
            comment = class_data['comment']
            subClassOf_list = class_data['subClassOf']

            # precompute optional fstrings
            subClassOf_rdf_lines = [
                f'    <rdfs:subClassOf rdf:resource="{sc}"/>' for sc in subClassOf_list if sc]
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

    def get_properties(self,
                       subject: URIRef,
                       properties_to_get: dict
                       ) -> dict[str, URIRef]:
        """
        Get the given properties for the given entity in the graph
        """
        properties = dict()
        for property_name, property in properties_to_get.items():
            property_values = list(self.graph.objects(subject, property))
            properties[property_name] = property_values
        return properties

    @staticmethod
    def _get_uri_from_embedding_string(embedding_string: str):
        """Quickly grab the URI from an embedding string"""
        pattern = re.compile(r'<rdf:Description rdf:about="((?:.)+?)">')
        uri = pattern.match(embedding_string).group(1)
        return uri

    def _get_properties_from_embedding_strings(self,
                                               embedding_strings: list[str],
                                               properties_to_get: dict[str,
                                                                       URIRef] = None
                                               ) -> dict[URIRef, dict[str, URIRef]]:
        """
        Convenience for first getting the URI from the embedding
        strings, and then getting the given properties for them in the
        graph.
        """
        subject_properties = {}
        for idx, embedding_string in enumerate(embedding_strings):

            # Grab the URI
            uri = self._get_uri_from_embedding_string(embedding_string)
            reference = URIRef(uri)

            if properties_to_get:
                properties = self.get_properties(
                    reference,
                    properties_to_get
                )
                properties['priority'] = idx
                subject_properties[reference] = properties
            else:
                subject_properties[reference] = None
        return subject_properties

    def get_parent_classes(self,
                           target_class: URIRef
                           ) -> list[URIRef]:
        """Get all direct parents of the given class"""
        parents = list(self.graph.objects(target_class, RDFS.subClassOf))
        return parents

    def get_all_parent_classes(self,
                               target_class: URIRef,
                               max_depth: int = 4
                               ) -> list[URIRef]:
        """Get all the parent classes in the hierarchy"""

        parents = list()
        nodes_to_check = queue.Queue()
        nodes_to_check.put(target_class)
        depth = 0

        while not nodes_to_check.empty():
            if depth > max_depth:
                #print(f'Maximum recursion depth reached ({max_depth})')
                break

            # get the next node to check from the queue
            node_to_check = nodes_to_check.get()

            # Add parent classes to the queue and parents list
            for parent in self.get_parent_classes(node_to_check):
                nodes_to_check.put(parent)
                parents.append(parent)

            depth += 1

        return list(set(parents))
