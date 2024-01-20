# see Conversation Knowledge Graph: https://python.langchain.com/docs/modules/memory/types/kg
# see source code: https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/memory/kg.py

# see NetworkX Graph: https://python.langchain.com/docs/use_cases/graph/graph_networkx_qa
from os import path
from typing import Any, Dict, List, Type, Union
import time


from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.messages import get_buffer_string


from langchain.chains.llm import LLMChain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.memory.utils import get_prompt_input_key

# from langchain_community.vectorstores import Chroma
from langchain_community.graphs import RdfGraph

from server.memory.prompts import (
    ENTITY_EXTRACTION_PROMPT,
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
)

from server.memory.semantic.learn import memorize


class SemanticLongTermMemory(BaseChatMemory):
    """Knowledge graph conversation memory.

    Integrates with external knowledge graph to store and retrieve
    information about knowledge triples in the conversation.
    """

    k: int = 2
    human_prefix: str = "Human"
    ai_prefix: str = "AI"
    kg_path: str = "database/_memories/knowledge.ttl"
    graph: RdfGraph = RdfGraph(kg_path)

    knowledge_extraction_prompt: BasePromptTemplate = KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT
    entity_extraction_prompt: BasePromptTemplate = ENTITY_EXTRACTION_PROMPT
    llm: BaseLanguageModel

    memory_key: str = "memories"  #: :meta private:

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history buffer."""

        recent_messages = self.chat_memory.messages[-self.k:]
        user_input = inputs[self.input_key]

        # context_chunk = "".join((
        #     "LAST MESSAGES:\n",
        #     str(recent_messages),
        #     "\n\n CURRENT USER MESSAGE:\n",
        #     user_input
        # ))
        # print(context_chunk)

        # Do named entity recognition on the last k messages.
        entities = self._get_current_entities(inputs)
        print(entities)

        # Get all knowledge about the entities from the memory knowledge
        # graph
        data = dict()
        for entity in entities:
            knowledge = get_entity_knowledge(entity)
            if knowledge:
                data[entity] = f"On {entity}: {'. '.join(knowledge)}."

        # Filter the knowledge to only relevant information to the last
        # k messages

        context = ''

        return {self.memory_key: context}
    
        # entities = self._get_current_entities(inputs)

        # summary_strings = []
        # for entity in entities:
        #     knowledge = self.kg.get_entity_knowledge(entity)
        #     if knowledge:
        #         summary = f"On {entity}: {'. '.join(knowledge)}."
        #         summary_strings.append(summary)
        # context: Union[str, List]
        # if not summary_strings:
        #     context = [] if self.return_messages else ""
        # elif self.return_messages:
        #     context = [
        #         self.summary_message_cls(content=text) for text in summary_strings
        #     ]
        # else:
        #     context = "\n".join(summary_strings)

        # return {self.memory_key: context}

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

    # def get_knowledge_triplets(self, input_string: str) -> List[KnowledgeTriple]:
    #     chain = LLMChain(llm=self.llm, prompt=self.knowledge_extraction_prompt)
    #     buffer_string = get_buffer_string(
    #         self.chat_memory.messages[-self.k * 2 :],
    #         human_prefix=self.human_prefix,
    #         ai_prefix=self.ai_prefix,
    #     )
    #     output = chain.predict(
    #         history=buffer_string,
    #         input=input_string,
    #         verbose=True,
    #     )
    #     knowledge = parse_triples(output)
    #     return knowledge

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs, outputs)
        # self._get_and_update_kg(inputs)
        # self.kg.write_to_gml("knowledge.gml")

    def clear(self) -> None:
        """Clear memory contents."""
        super().clear()
        pass

    def memorize(self, conversation_history: str):
        print(self.chat_memory)
        print("Memorize not implemented")
        #memorize(conversation_history, self.llm, self.tbox_db)


def get_entities(entity_str: str) -> List[str]:
    """Extract entities from entity string."""
    if entity_str.strip() == "NONE":
        return []
    else:
        return [w.strip() for w in entity_str.split(",")]
