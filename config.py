"""Configuration file for the semantic memory system"""

import os


# List of ontologies T-Box which will serve as our conversational agent's
# world model
ONTOLOGIES_PATHS = [
    'ontologies/dbpedia.owl',  # General ontology
    'ontologies/foaf.owl'  # People ontology
    # 'http://xmlns.com/foaf/spec/index.rdf'  # People ontology
    # 'https://www.cirma.unito.it/drammar/drammar.owl' # Emotions ontology
]

# Base knowledge used to populate the memory on first time use
BASE_KNOWLEDGE_PATH = './ontologies/base_knowledge.ttl'

# Path to ontology vector databases directory
_ONTOLOGY_DBS_PATH = './database/vector_db/'

# Path to memories directory
_MEMORIES_PATH = './database/_memories/'

# Paths of the vector databases we'll need
CLASS_DB_PATH = _ONTOLOGY_DBS_PATH + 'oa_predicates_db'
PREDICATES_DB_PATH = _ONTOLOGY_DBS_PATH + 'oa_classes_db'
ENTITIES_DB_PATH = _MEMORIES_PATH + 'entities_db'

# Path of RDF memory graph
MEMORY_PATH = _MEMORIES_PATH + 'knowledge.ttl'


# Create ontology databases directory if it doesn't exist
if not os.path.exists(_ONTOLOGY_DBS_PATH):
    os.makedirs(_ONTOLOGY_DBS_PATH)

# Create memories directory if it doesn't exist
if not os.path.exists(_MEMORIES_PATH):
    os.makedirs(_MEMORIES_PATH)