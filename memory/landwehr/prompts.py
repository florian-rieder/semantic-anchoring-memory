from langchain.prompts import PromptTemplate


_QUERY_CREATION_TEMPLATE = (
    "Chat history (for reference only): "
    "{history}\n\n"
    "Task:\n"
    "Create a search query for a similarity search in the AI memory that helps answer the user's last message."
    " You cannot ask for clarification. Provide only the query."
    "Query: "
)

QUERY_CREATION_PROMPT = PromptTemplate(
    input_variables=['history'],
    template=_QUERY_CREATION_TEMPLATE
)


_FACT_EXTRACTION_TEMPLATE = (
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
    template=_FACT_EXTRACTION_TEMPLATE
)


_SUMMARIZER_PROMPT_TEMPLATE = (
    "Generate a concise summary of the conversation transcript, focusing on key"
    " facts and memorable details related to the user's life."
    " Write sentences which are understandable in isolation. Always refer to named entities by their name. Prioritise proper name over generic names."
    " Always refer to the user as User, and to the AI as Assistant."
    " Highlight significant events, achievements, personal preferences, and any"
    " noteworthy information that provides a comprehensive overview of the user's experiences and interests:\n\n"
    "Conversation history:\n\n"
    "{text}"
    "\n\nSummary of the transcript:\n"
)

SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=['text'],
    template=_SUMMARIZER_PROMPT_TEMPLATE,
)


_CHUNK_SUMMARIZER_PROMPT_TEMPLATE = (
    'Generate a concise summary of the given chunk of conversation transcript, focusing on key'
    " facts and memorable details related to the user's life and the conversation topic."
    'Summary of previous chunks (for reference only):\n'
    '{summary_of_previous_chunks}'
    'Chunk to summarize'
    '{chunk}'
)

CHUNK_SUMMARIZER_PROMPT = PromptTemplate(
    input_variables=['summary_of_previous_chunks', 'chunk'],
    template=_CHUNK_SUMMARIZER_PROMPT_TEMPLATE,
)


# From Landwehr et al. 2023
_RESPONSE_GENERATION_TEMPLATE = (
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
    template=_RESPONSE_GENERATION_TEMPLATE
)
