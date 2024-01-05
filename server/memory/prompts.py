"""
Prompts
"""
from langchain.prompts import PromptTemplate


FACT_EXTRACTION_TEMPLATE = (
    "Extract all important observations about people, places and"
    " concepts from the given chunk."
    " Make sure that each fact is understandable in isolation."
    " Always refer to entities by their name. Proper noun is preferred."
    " For example, when referring to the user, user 'the user' instead of 'they'."
    " Don't repeat facts from the context, only from the current chunk. Output each fact on a new line."
    " If there is nothing important to remember, like if the user is just greeting, output nothing.\n\n"
    "Context (for reference only):\n"
    "{summary}\n\n"
    "Current chunk (for fact extraction):\n"
    "{chunk}\n\n"
    "Important facts from the current chunk:"
)

FACT_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=['summary', 'chunk'],
    template=FACT_EXTRACTION_TEMPLATE
)


CONVERSATION_SUMMARY_TRIPLET_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence tasked with extracting knowledge triples"
    " from a conversation summary. Ensure consistency in named entities, always using 'User' for the user."
    " A knowledge triple consists of a subject, a predicate, and an object. The subject is the entity being described,"
    " the predicate is the property of the subject being described, and the object is the value of the property."
    " If the predicate is an ObjectProperty, indicate it with 'O:', and if it's a DataProperty, use 'D:' in the output."
    "\n\nExamples:\n"
    "1. The user, named Florian, is a student.\n   Output: (O: User, is, O: student), (O: User, is named, D: Florian)"
    "\n2. The user, a student, received a job offer to work on a website in December.\n"
    "   Output: (O: User, received, O: job offer), (O: job offer, concerns the task of, D: working on a website), (O: job offer, received in, D: December)"
    "\n3. The user, a student, declined a job offer due to a busy schedule at the end of the semester.\n"
    "   Output: (O: User, declined, O: job offer), (O: User, has, O: busy schedule), (O: busy schedule, is during time period, D: end of semester)"
    "\n4. Paris is the capital of France. The user went to Paris in May 2022.\n"
    "   Output: (O: Paris, is capital, O: France), (O: Paris, is, O: city), (O: User, has traveled to , O: Paris)"
    "\nSummary of the last conversation:\n"
    "{summary}"
    "\nOutput:"
)


NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=CONVERSATION_SUMMARY_TRIPLET_EXTRACTION_TEMPLATE,
)


KG_TRIPLE_DELIMITER = '<|>'
TRIPLET_EXTRACTION_TEMPLATE = (
    "You are a networked intelligence helping a human track knowledge triples"
    " about all relevant people, things, concepts, etc. and integrating"
    " them with your knowledge stored within your weights"
    " as well as that stored in a knowledge graph."
    " Extract all of the knowledge triples from the last line of conversation."
    " A knowledge triple is a clause that contains a subject, a predicate,"
    " and an object. The subject is the entity being described,"
    " the predicate is the property of the subject that is being"
    " described, and the object is the value of the property."
    " When information about the user is to be recorded, always use 'User' as the entity name.\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "User: Hello, my name is Gemini !\n"
    "AI: Hello Gemini! It's nice to meet you. I'm an AI assistant trained"
    " to help with various topics. How can I assist you today?\n"
    f"Output: (User, is named, Gemini)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "User: Did you hear aliens landed in Area 51?\n"
    "AI: No, I didn't hear that. What do you know about Area 51?\n"
    "User: It's a secret military base in Nevada.\n"
    "AI: What do you know about Nevada?\n"
    "Last line of conversation:\n"
    "User: It's a state in the US. It's also the number 1 producer of gold in the US.\n\n"
    f"Output: (Nevada, is a, state){KG_TRIPLE_DELIMITER}(Nevada, is in, US)"
    f"{KG_TRIPLE_DELIMITER}(Nevada, is the number 1 producer of, gold)\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "User: Hello.\n"
    "AI: Hi! How are you?\n"
    "User: I'm good. How are you?\n"
    "AI: I'm good too.\n"
    "Last line of conversation:\n"
    "User: I'm going to the store.\n\n"
    "Output: NONE\n"
    "END OF EXAMPLE\n\n"
    "EXAMPLE\n"
    "Conversation history:\n"
    "User: What do you know about Descartes?\n"
    "AI: Descartes was a French philosopher, mathematician, and scientist who lived in the 17th century.\n"
    "User: The Descartes I'm referring to is a standup comedian and interior designer from Montreal.\n"
    "AI: Oh yes, He is a comedian and an interior designer. He has been in the industry for 30 years. His favorite food is baked bean pie.\n"
    "Last line of conversation:\n"
    "User: Oh huh. I know Descartes likes to drive antique scooters and play the mandolin.\n"
    f"Output: (Descartes, likes to drive, antique scooters){KG_TRIPLE_DELIMITER}(Descartes, plays, mandolin)\n"
    "END OF EXAMPLE\n\n"
    "Conversation history (for reference only):\n"
    "{history}"
    "\nLast line of conversation (for extraction):\n"
    "Human: {input}\n\n"
    "Output:"
)
KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=TRIPLET_EXTRACTION_TEMPLATE,
)


_DEFAULT_ENTITY_EXTRACTION_TEMPLATE = """You are an AI assistant reading the transcript of a conversation between an AI and a human. Extract all of the proper nouns from the last line of conversation. As a guideline, a proper noun is generally capitalized. You should definitely extract all names and places. "I" usually refers to the User.

The conversation history is provided just in case of a coreference (e.g. "What do you know about him" where "him" is defined in a previous line) -- ignore items mentioned there that are not in the last line.

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

Conversation history (for reference only):
{history}
Last line of conversation (for extraction):
User: {input}

Output:"""
ENTITY_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=_DEFAULT_ENTITY_EXTRACTION_TEMPLATE
)


QUERY_CREATION_TEMPLATE = (
    "Chat history (for reference only): "
    "{history}\n\n"
    "Task:\n"
    "Create a search query for a similarity search in the AI memory that helps answer the user's last message."
    " You cannot ask for clarification. Provide only the query."
    "Query: "
)

QUERY_CREATION_PROMPT = PromptTemplate(
    input_variables=['history'],
    template=QUERY_CREATION_TEMPLATE
)


RESPONSE_GENERATION_TEMPLATE = (
    "You are acting as a virtual character and you are having a conversation with a user."
    " The character you are simulating is named {name}."
    " Your task is to answer the user based on the chat history."
    " You should answer the last message in the chat history.\n\n"
    "CHARACTER_BIO_START\n\n"
    "{bio}\n\n"
    "CHARACTER_BIO_END\n\n"
    "This is the current chat history:\n"
    "START_CHAT_HISTORY\n\n"
    "Most recent messages:"
    "{history}\n\n"
    "END_CHAT_HISTORY\n\n"
    "The simulated character has memories. Use the memories to guide your answer.\n"
    "START_MEMORIES\n\n"
    "{memories}\n\n"
    "END MEMORIES\n\n"
    "The answer must be based on the memories."
    " Do not talk about anything that is not in the memories."
    " For each sentence , provide a source like [MEMORY_i]."
    " Create a character response to the last message of the user."
    " The response must be from the point of view of Sherlock Holmes."
    " The response should be around 50-70 words.\n\n"
    "Response:"
)

RESPONSE_GENERATION_PROMPT = PromptTemplate(
    input_variables=["name", "bio", "history", "memories"],
    template=RESPONSE_GENERATION_TEMPLATE
)
