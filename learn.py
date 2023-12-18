"""
Responsibility:
    Converts raw text into condensed memories for virtual AI characters.
Process:
    Splits the source text into smaller chunks, processes each chunk
    using the LLM to extract important facts, and creates corresponding
    memories.
"""
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import LLMChain
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate


from prompts import KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT, FACT_EXTRACTION_PROMPT



def split_chunk_context_pairs(text: str, llm: BaseLanguageModel) -> List[tuple]:
    """Split text into multiple components."""
    # 1. Split source text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, chunk_overlap=0)
    chunks = text_splitter.split_text(text)

    # 2. Process each chunk in isolation but with a context that contains a
    # summary of all previous chunks.
    contexts = []
    for idx, chunk in enumerate(chunks):
        chunks_before = chunks[:idx]

        context = "".join(chunks_before)

        summary = summarize(context, llm)

        context.append((chunk, summary))

    return contexts


def summarize(text: str, num_paragraphs: int, llm: BaseLanguageModel) -> str:
    # Define summarizer prompt
    summarizer_prompt_template = (
        "Generate a concise summary of the conversation transcript, focusing on key"
        " facts and memorable details related to the user's life."
        " Highlight significant events, achievements, personal preferences, and any"
        " noteworthy information that provides a comprehensive overview of the user's experiences and interests:\n\n"
        "{previous_chunks}"
    )
    summarizer_prompt = PromptTemplate(
        input_variables=['text', 'num_paragraphs'],
        template=summarizer_prompt_template,
    )

    chain = summarizer_prompt | llm

    response = chain.predict(
        text=text,
        num_paragraphs=num_paragraphs
    )

    # TODO: Output validation and parsing

    return response


def extract_facts(chunk: str, summary: str, llm: BaseLanguageModel) -> List[str]:
    # Define fact extraction prompt
    chain = LLMChain(llm=llm, prompt=FACT_EXTRACTION_PROMPT)


    response = chain.predict(chunk=chunk, summary=summary)

    # TODO: Parse output

    return response

def extract_triplets(last_line: str, history: str, llm: BaseLanguageModel):
    chain = LLMChain(
        prompt=KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
        llm=llm,
    )

    triplets = chain.predict(input=last_line, history=history)

    return triplets


if __name__ == "__main__":
    from langchain.llms import OpenAI

    test_text = """
    Le charbon est une roche sédimentaire combustible, riche en carbone, de couleur noire ou marron foncé, formée à partir de la dégradation partielle de la matière organique des végétaux. Il est exploité dans des mines, appelées charbonnages.

Couvrant 26,8 % des besoins énergétiques mondiaux en 2020, le charbon est la seconde ressource énergétique de l'humanité, derrière le pétrole (29,5 %) et devant le gaz naturel (23,7 %), et la première source d'électricité avec 35,2 % de la production d'électricité en 2020 contre 38,3 % en 1973.

La consommation mondiale est concentrée en 2022 à 73,3 % dans trois pays : Chine 54,8 %, Inde 12,4 % et États-Unis 6,1 %. Elle atteint une tonne de charbon/an/habitant.L'AIE prévoit que la consommation mondiale devrait être stable entre 2022 et 2025, la baisse de la consommation en Europe et Amérique du nord étant compensée par son augmentation en Inde et en Asie du Sud-Est.

Souvent appelé houille, il était autrefois appelé charbon de terre en opposition au charbon de bois.

Au cours de plusieurs millions d'années, l'accumulation et la sédimentation de débris végétaux dans un environnement de type tourbière provoque une modification graduelle des conditions de température, de pression et d'oxydo-réduction dans la couche de charbon qui conduit, par carbonisation, à la formation de composés de plus en plus riches en carbone : la tourbe (moins de 50 %), le lignite (50 à 60 %), la houille (60 à 90 %) et l'anthracite (93 à 97 %). La formation des plus importants gisements de charbon commence au Carbonifère, de −360 à −295 Ma.

Les réserves mondiales de charbon sont estimées à 22 436 EJ (exajoules) fin 2020, dont 25,8 % aux États-Unis, 15,5 % en Chine, 12,5 % en Russie, 12,1 % en Australie et 12,0 % en Inde, soit 129 ans de production au rythme de 2022 ; cette production est concentrée à 85,4 % dans six pays : la Chine (50,6 %), l'Inde (10,0 %), l'Indonésie (7,5 %), les États-Unis (6,4 %), l'Australie (5,6 %) et la Russie (5,3 %) ; elle a progressé de 85 % en 32 ans (1990-2022) malgré des baisses en 2015-2016 (-8,3 %) et en 2020, où la crise liée à la pandémie de Covid-19 l'a fait chuter de 4,7 %.

Son extraction a rendu possible la révolution industrielle aux XVIIIe et XIXe siècles. Sa combustion engendre 44,0 % des émissions de CO2 dues à l'énergie en 2019, contre 33,7 % pour le pétrole et 21,6 % pour le gaz naturel. Pour atteindre l'objectif des négociations internationales sur le climat de maintenir la hausse des températures en deçà de 2 °C par rapport à l'ère préindustrielle, il faudrait globalement s'abstenir d'extraire plus de 80 % du charbon disponible dans le sous-sol mondial, d'ici à 2050. 
    """

    llm = OpenAI(
        model='gpt-3.5-turbo-instruct',
        temperature=0.05
    )

    #split_chunk_context_pairs(test_text, llm)
    print(extract_triplets('Hello, my name is Florian. How are you today ?', '', llm))
