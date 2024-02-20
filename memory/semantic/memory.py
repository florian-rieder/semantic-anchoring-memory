"""
This module defines the SemanticLongTermMemory class, a specialized form
of chat memory for a Language Learning Model (LLM). 

The SemanticLongTermMemory class integrates with an external knowledge
graph to store and retrieve information about knowledge triples in the
conversation. It uses the BaseChatMemory as a base class and extends it
with functionality specific to handling semantic long-term memory. The
memorization process is only launched at the end of a conversation, as
opposed to after each message, for the last couple of messages. The
latter option wasn't chosen because it would introduce a large latency
between messages, as the memory creation process is relatively intensive
in its current form.

It also includes a helper function `get_entities` to extract entities
from a given entity string.

Classes
-------
SemanticLongTermMemory
    A long term chat memory for a LLM that integrates with an external
    knowledge graph (the A-box, which is the memory storage).

Functions
---------
get_entities(entity_str: str) -> List[str]
    Extracts entities from a given entity string.
"""

from typing import Any, Dict, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.messages import get_buffer_string

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key

from memory.semantic.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)
from memory.semantic.learn import memorize
from memory.semantic.store import SemanticStore


class SemanticLongTermMemory(BaseChatMemory):
    """
    A subclass of BaseChatMemory that integrates with an external
    knowledge graph to store and retrieve information about knowledge
    triples in the conversation. It uses a language model (llm) and a
    semantic store to process and store the information.

    Attributes
    ----------
    semantic_store: SemanticStore
        An instance of SemanticStore used to encode and store knowledge
        triples.
    llm: BaseLanguageModel
        An instance of a language model used to process and understand
        the conversation.
    knowledge_extraction_prompt: BasePromptTemplate
        A template for extracting knowledge triples from the conversation.
    entity_extraction_prompt: BasePromptTemplate
        A template for extracting entities from the conversation.
    k: int
        The number of last messages to consider for entity recognition.
    human_prefix: str
        The prefix used to denote human messages in the conversation.
    ai_prefix: str
        The prefix used to denote AI messages in the conversation.
    memory_key: str
        The key used to store memory variables.

    Methods
    -------
    load_memory_variables(inputs: Dict[str, Any]) -> Dict[str, Any]:
        Returns the history buffer after performing named entity
        recognition on the last k messages.
    memory_variables() -> List[str]:
        Returns a list of memory variables.
    _get_prompt_input_key(inputs: Dict[str, Any]) -> str:
        Returns the input key for the prompt.
    _get_prompt_output_key(outputs: Dict[str, Any]) -> str:
        Returns the output key for the prompt.
    get_current_entities(input_string: str) -> List[str]:
        Returns the current entities in the conversation.
    """

    semantic_store: SemanticStore
    llm: BaseLanguageModel

    knowledge_extraction_prompt: BasePromptTemplate = NEW_KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT

    k: int = 4
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "memories"  #: :meta private:

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""

        # Do named entity recognition on the last k messages.
        entities = self._get_current_entities(inputs)

        print('Entities found in last k messages:')
        print(entities)

        # Get all knowledge about the entities from the memory knowledge
        # graph
        summary_strings = list()
        for entity in entities:
            knowledge = self.semantic_store.abox.get_entity_knowledge(entity)
            if knowledge:
                entity_knowledge_summary = f"On {entity}: {'. '.join(knowledge)}."
                summary_strings.append(entity_knowledge_summary)

        context = "\n".join(summary_strings)

        return {self.memory_key: context}

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def _get_prompt_input_key(self, inputs: Dict[str, Any]) -> str:
        """Get the input key for the prompt."""
        if self.input_key is None:
            return get_prompt_input_key(inputs, self.memory_variables)
        return self.input_key

    def _get_prompt_output_key(self, outputs: Dict[str, Any]) -> str:
        """Get the output key for the prompt."""
        if self.output_key is None:
            if len(outputs) != 1:
                raise ValueError(
                    f"One output key expected, got {outputs.keys()}")
            return list(outputs.keys())[0]
        return self.output_key

    def get_current_entities(self, input_string: str) -> List[str]:
        """Get the current entities in the conversation."""
        chain = LLMChain(
            llm=self.llm,
            prompt=self.entity_extraction_prompt
        )

        # Get the chat history as a string
        buffer_string = get_buffer_string(
            self.chat_memory.messages[-self.k * 2:],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        # Use LLM to determine entities relevant to the last message
        output = chain.predict(
            history=buffer_string,
            input=input_string,
        )

        return get_entities(output)

    def _get_current_entities(self, inputs: Dict[str, Any]) -> List[str]:
        """Get the current entities in the conversation."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        return self.get_current_entities(inputs[prompt_input_key])

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        #self.semantic_store.abox.save_graph()

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()

    def memorize(self, conversation_history: str):
        """Memorize the conversation. To be called at the end of the
        conversation."""
        print(self.chat_memory)
        print('Start memorization')
        memorize(conversation_history, self.llm, self.semantic_store)
        print('End memorization')


def get_entities(entity_str: str) -> List[str]:
    """Extract entities from entity string."""
    if entity_str.strip() == "NONE":
        return []
    else:
        return [w.strip() for w in entity_str.split(",")]
