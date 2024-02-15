"""
Prompts
"""
from langchain.prompts import PromptTemplate

_CHUNK_SUMMARIZER_TEMPLATE = (
    'Generate a concise summary of the given chunk of conversation transcript, focusing on key'
    " facts and memorable details related to the user's life and the conversation topic."
    'Summary of previous chunks (for reference only):\n'
    '{summary_of_previous_chunks}'
    'Chunk to summarize'
    '{chunk}'
)

CHUNK_SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=['summary_of_previous_chunks', 'chunk'],
    template=_CHUNK_SUMMARIZER_TEMPLATE,
)


# Define summarizer prompt
_SUMMARIZER_TEMPLATE = """
You are an AI memory entity tasked with extracting key insights and notable details from the provided text
to capture the most important information to remember about the topic.
Craft concise sentences focusing on significant facts, events, achievements,
preferences, and memorable aspects of the subject's experiences and about the topic of the text.
Ensure these sentences are understandable in isolation.
Prioritize unique and impactful information over mundane details.
Always refer to named entities by their name, emphasizing proper names over generics.
When the text is authored by a specific individual, refer to them as "Author".
In a conversation, address the subject as "User" and the AI as "Assistant".

Input text:

{text}


Summary of the Input:

"""

SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=['text'],
    template=_SUMMARIZER_TEMPLATE,
)


KG_TRIPLE_DELIMITER = '|'
_CONVERSATION_SUMMARY_TRIPLET_EXTRACTION_TEMPLATE = (
"""You are a networked intelligence tasked with extracting knowledge triples from a summary.
Ensure consistency in named entities.
When naming the entity who is the locutor, use 'User' for the user in a conversation, or 'Author' in a written text.
A knowledge triple consists of a subject, a predicate, and an object.
The subject is the entity being described, the predicate is the property of the subject being described, and the object is the value of the property.
Complex semantics may require intermediary nodes (also called blank nodes), that allow for multiple properties to be assigned to one concept.
"""
f"Triples must be in the form (subject{KG_TRIPLE_DELIMITER} predicate{KG_TRIPLE_DELIMITER} object)"
"""
EXAMPLE
Summary:
Paris is the capital of France. The user went to Paris in May 2022.
Output:
"""
f"(Paris{KG_TRIPLE_DELIMITER} is capital of{KG_TRIPLE_DELIMITER} France), (Paris{KG_TRIPLE_DELIMITER} is{KG_TRIPLE_DELIMITER} city), (User{KG_TRIPLE_DELIMITER} has traveled{KG_TRIPLE_DELIMITER} ParisTravel), (ParisTravel{KG_TRIPLE_DELIMITER} took place in{KG_TRIPLE_DELIMITER} May 2022), (ParisTravel{KG_TRIPLE_DELIMITER} destination{KG_TRIPLE_DELIMITER} Paris)"
"""
EXAMPLE
Summary:
Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.
Output:
"""
f"(Descartes{KG_TRIPLE_DELIMITER} country of origin{KG_TRIPLE_DELIMITER} France), (Descartes{KG_TRIPLE_DELIMITER} is a{KG_TRIPLE_DELIMITER} philosopher){KG_TRIPLE_DELIMITER} (Descartes{KG_TRIPLE_DELIMITER} is a{KG_TRIPLE_DELIMITER} mathematician){KG_TRIPLE_DELIMITER} (Descartes{KG_TRIPLE_DELIMITER} is a{KG_TRIPLE_DELIMITER} scientist){KG_TRIPLE_DELIMITER} (Descartes{KG_TRIPLE_DELIMITER} lived in{KG_TRIPLE_DELIMITER} the 17th century)"
"""
END OF EXAMPLE


Summary:
{summary}

Output:

"""
)

NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["summary"],
    template=_CONVERSATION_SUMMARY_TRIPLET_EXTRACTION_TEMPLATE,
)


_ENTITY_EXTRACTION_TEMPLATE = """You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the proper nouns from the last line of conversation. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places. "I" usually refers to the User.

Include all entities relevant to the last line of conversation.

Return the output as a single comma-separated list, or NONE if there is nothing of note to return (e.g. the user is just issuing a greeting or having a simple conversation).

EXAMPLE
Conversation history:
User: how's it going today?
AI: "It's going great! How about you?"
User: good! busy working on Langchain. lots to do.
AI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"
Last line:
User: I'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff.
Output: User, Langchain
END OF EXAMPLE

EXAMPLE
Conversation history:
User: how's it going today?
AI: "It's going great! How about you?"
User: good! busy working on Langchain. lots to do.
AI: "That sounds like a lot of work! What kind of things are you doing to make Langchain better?"
Last line:
User: i'm trying to improve Langchain's interfaces, the UX, its integrations with various products the user might want ... a lot of stuff. I'm working with Person #2.
Output: User, Langchain, Person #2
END OF EXAMPLE

EXAMPLE
User: Who was the first president of the united states ?
AI: The first president of the United States was George Washington. He served from 1789 to 1797.\n"
Last line:
User: Who succeded him ?
Output: User, United States, George Washington
END OF EXAMPLE

Conversation history (for reference only):
{history}
Last line of conversation (for extraction):
User: {input}

Output:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=_ENTITY_EXTRACTION_TEMPLATE
)




# Unused
# See 
_TRIPLET_ENCODER_TEMPLATE = (
    """You are an expert agent specialized in analyzing product specifications in an online retail store.
Your task is to identify the entities and relations requested with the user prompt, from a given product specification.
You must generate the output in a JSON containing a list with JOSN objects having the following keys: "head", "head_type", "relation", "tail", and "tail_type".
The "head" key must contain the text of the extracted entity with one of the types from the provided list in the user prompt, the "head_type"
key must contain the type of the extracted head entity which must be one of the types from the provided user list,
the "relation" key must contain the type of relation between the "head" and the "tail", the "tail" key must represent the text of an
extracted entity which is the tail of the relation, and the "tail_type" key must contain the type of the tail entity. Attempt to extract as
many entities and relations as you can.\n
"""
"Based on the following example, extract entities and relations from the provided text.\n"


"--> Beginning of example\n"

"# Information\n"
"""
The user is named Florian. He has a cat.
"""

"\n# Output\n"
"""[
  {{
    "head": "User",
    "head_type": "Person",
    "relation": "isNamed",
    "tail": "Florian",
    "tail_type": "literal"
  }},
  {{
    "head": "User",
    "head_type": "Person",
    "relation": "hasPet",
    "tail": "cat",
    "tail_type": "Animal"
  }}
]
"""
"--> End of example\n\n"

"Use the following entity types:"

"# ENTITY TYPES:\n"
"{entity_types}\n\n"

"Use the following relation types:\n"
"{relation_types}\n\n"

"You can also use regular OWL relations and entity types.\n\n"

"For the following information, extract entitites and relations as in the provided example.\n\n"
"# Information\n"
"{information}\n"
"# Output"
)

TRIPLET_ENCODER_PROMPT = PromptTemplate(
    input_variables=["entity_types", "relation_types", "information"],
    template=_TRIPLET_ENCODER_TEMPLATE
)


_CHOOSE_PREDICATE_TEMPLATE = """Choose the predicate from the list which corresponds best to the given intent.
You can also use predicates from common namespaces present in your weights: RDF, RDFS, OWL, SKOS, FOAF.
If no predicate in the list fits, create your own.
If you create a new predicate, only use the namespace http://example.com/.
Only output the chosen predicate and nothing else.

List of predicates:
{predicates}

Intent:
{intent}

Chosen predicate:

"""

CHOOSE_PREDICATE_PROMPT = PromptTemplate(
    input_variables=['predicates', 'intent'],
    template = _CHOOSE_PREDICATE_TEMPLATE
)

_CHOOSE_CLASS_TEMPLATE = """Choose the class from the list which corresponds best to the given intent.
Additionnally, you can use "Literal" (which represents simply a string of test, not to be used on subjects, but can be used on objects in some cases), and "http://www.w3.org/2002/07/owl#Thing".
Sometimes, if you can't describe the class using the provided classes, it's better to use Literal or Thing.
You can use common classes from common namespaces: RDF, RDFS, OWL, SKOS.

EXAMPLE
List of classes:
http://dbpedia.org/ontology/Animal
http://dbpedia.org/ontology/Eukaryote
http://www.w3.org/2002/07/owl#Thing
Literal

Intent:
Get the correct class for object "a cat" in the triple (User, has, a cat)

Chosen class:
http://dbpedia.org/ontology/Animal

END OF EXAMPLE

EXAMPLE
List of classes:
http://dbpedia.org/ontology/Scientist
http://dbpedia.org/ontology/Person
http://dbpedia.org/ontology/Animal
http://dbpedia.org/ontology/Eukaryote
http://www.w3.org/2002/07/owl#Thing

Intent:
Get the correct class for subject "User" in the triple (User, has, a cat)

Chosen class:
http://dbpedia.org/ontology/Person

END OF EXAMPLE

EXAMPLE
List of classes:
http://dbpedia.org/ontology/Scientist
http://dbpedia.org/ontology/Person
http://dbpedia.org/ontology/Animal
http://dbpedia.org/ontology/Eukaryote
http://www.w3.org/2002/07/owl#Thing
Literal

Intent:
Get the correct class for object "Florian" in the triple (User, is named, Florian)

Chosen class:
Literal

END OF EXAMPLE

List of classes:
{classes}

Intent:
{intent}

Chosen class:

"""

CHOOSE_CLASS_PROMPT = PromptTemplate(
    input_variables=['classes', 'intent'],
    template = _CHOOSE_CLASS_TEMPLATE
)