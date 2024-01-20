from os import path
from typing import Any, Dict, List
from urllib.parse import urlparse

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.messages import get_buffer_string

from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key

from server.memory.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)
from server.memory.semantic.learn import memorize
from server.memory.semantic.store import SemanticStore


class SemanticLongTermMemory(BaseChatMemory):
    """Knowledge graph conversation memory.

    Integrates with external knowledge graph to store and retrieve
    information about knowledge triples in the conversation.
    """

    semantic_store: SemanticStore
    llm: BaseLanguageModel

    knowledge_extraction_prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT

    k: int = 4
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    memory_key: str = "memories"  #: :meta private:

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""

        # recent_messages = self.chat_memory.messages[-self.k:]
        # user_input = inputs[self.input_key]

        # Do named entity recognition on the last k messages.
        entities = self._get_current_entities(inputs)
        print(entities)

        # Get all knowledge about the entities from the memory knowledge
        # graph
        summary_strings = list()
        for entity in entities:
            knowledge = self.get_entity_knowledge(entity)
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

        buffer_string = get_buffer_string(
            self.chat_memory.messages[-self.k * 2:],
            human_prefix=self.human_prefix,
            ai_prefix=self.ai_prefix,
        )

        output = chain.predict(
            history=buffer_string,
            input=input_string,
        )

        return get_entities(output)

    def _get_current_entities(self, inputs: Dict[str, Any]) -> List[str]:
        """Get the current entities in the conversation."""
        prompt_input_key = self._get_prompt_input_key(inputs)
        return self.get_current_entities(inputs[prompt_input_key])

    def get_entity_knowledge(self, entity: str) -> list[str]:
        # Get similar entities using a similarity search in the entities database
        similar_entities = self.semantic_store.abox.query_entities(entity)
        entity_node = similar_entities[0]

        # Get all the knowledge about this entity
        # TODO: if more entities are revealed, gather knowledge about them also
        knowledge = list()
        for p, o in self.semantic_store.abox.graph.predicate_objects(entity_node):
            # Get the last bit of the URI
            # ex. https://example.com/Bob -> Bob
            p = urlparse(str(p)).path.split("/")[-1]
            o = urlparse(str(o)).path.split("/")[-1]

            knowledge_bit = f"{p} {o}"
            knowledge.append(knowledge_bit)

        # TODO: Filter knowledge to only what is relevant to the conversation

        return knowledge

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        # self._get_and_update_kg(inputs)
        # self.kg.write_to_gml("knowledge.gml")
        self.semantic_store.abox.save_graph()

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()

    def memorize(self, conversation_history: str):
        """Memorize the conversation. To be called at the end of the
        conversation."""
        print(self.chat_memory)
        memorize(conversation_history, self.llm, self.semantic_store)


def get_entities(entity_str: str) -> List[str]:
    """Extract entities from entity string."""
    if entity_str.strip() == "NONE":
        return []
    else:
        return [w.strip() for w in entity_str.split(",")]
