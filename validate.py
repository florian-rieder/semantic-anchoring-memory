"""
Summary
-------
Validation pipeline for the knowledge graph extraction

This script provides a validation pipeline for knowledge graph extraction.
It processes each topic directory, extracts information from text files,
generates knowledge graphs, computes statistics, and saves results.

Usage
-----
>>> python validate.py [-h] [--topics TOPICS [TOPICS ...]]

Arguments
---------
-t , --topics: Specify which folders inside validation/topics to
    process.
"""

import argparse
import json
import logging
import os
import shutil
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from rdflib import Graph

from config import EMBEDDING_MODEL_NAME
from ingest import init
from memory.semantic.learn import memorize as semantic_memorize
from memory.landwehr.landwehr import memorize as landwehr_memorize


logger = logging.getLogger(__name__)


def main(args):
    """
    Validation pipeline for the knowledge graph extraction.

    Arguments
    ---------
        args (argparse.Namespace): Command-line arguments.

    Returns
    -------
        None

    This function serves as the main entry point for the validation pipeline.
    It processes each topic directory, extracts information from text files,
    generates knowledge graphs, computes statistics, and saves results.

    The validation pipeline consists of the following steps:
    1. Iterate through each topic directory.
    2. Read the text file within the directory.
    3. Process the text file to generate a semantic knowledge graph.
    4. Compute statistics on the generated knowledge graph.
    5. Process the text using Landwehr to generate additional facts.
    6. Compute statistics on the generated memories
    7. Optionally retrieve information from DBpedia about the topic.
    8. Compute statistics on the DBpedia knowledge graph.
    9. Save statistics for each step to file.

    Note:
        This function assumes the existence of certain directory structures
        and file formats as per the validation pipeline's requirements.
    """
    # Define the directory containing the topic directories
    topics_dir = f'{os.getcwd()}/validation/topics/'

    if args.topics:
        # If topics have been defined with the command line argument,
        # only process the selected topics
        topic_directories = sorted(args.topics)
    else:
        # Get a list of all directories in the topics directory
        topic_directories = sorted(os.listdir(topics_dir))

    # 1. Loop through each topic directory
    for topic_directory in tqdm(topic_directories):
        print(f'Processing {topic_directory}...')

        # Construct full paths
        topic_dir_path = os.path.join(topics_dir, topic_directory)
        output_kg_path = os.path.join(topic_dir_path, 'output.ttl')
        entities_db_path = os.path.join(topic_dir_path, 'entities_db')
        stats_file_path = os.path.join(topic_dir_path, 'statistics.json')
        landwehr_db_path = os.path.join(
            topic_dir_path, 'landwehr_memories_db')
        landwehr_text_output_path = os.path.join(
            topic_dir_path, 'landwehr_memories.txt')
        dbpedia_resource_file_path = os.path.join(
            topic_dir_path, 'dbpedia.ttl')

        # Get metadata about the topic
        metadata = get_metadata(topic_dir_path)

        # Delete files and dbs we're going to overwrite
        if os.path.exists(output_kg_path):
            os.remove(output_kg_path)
        if os.path.exists(stats_file_path):
            os.remove(stats_file_path)
        if os.path.exists(entities_db_path):
            shutil.rmtree(entities_db_path)
        if os.path.exists(landwehr_db_path):
            shutil.rmtree(landwehr_db_path)

        # Initialize memory, using the same config as in the ingestion
        # script for consistency
        llm, store = init(
            # Save the output graph in the topic directory
            memory_path=output_kg_path,
            # Save the entities db associated to the graph in the topic
            # directory
            entities_db_path=entities_db_path,
            # Use no prior knowledge
            base_knowledge=None
        )

        # 2. Get the text file
        # Check if the path is a directory
        if not os.path.isdir(topic_dir_path):
            logger.warning(f'Not a valid directory {topic_dir_path}')
            continue

        # Call the function to read the first text file within the topic
        # directory
        text = read_first_text_file_in_dir(topic_dir_path)

        # 3. Process the text file and output a knowledge graph
        semantic_memorize(text, llm, store)

        # 4. Compute statistics over the resulting knowledge graph
        semantic_stats = get_statistics(output_kg_path)

        # 5. Process using Landwehr
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
        )

        landwehr_store = Chroma(
            persist_directory=landwehr_db_path,
            embedding_function=embeddings
        )

        landwehr_facts = landwehr_memorize(text, llm, landwehr_store)

        with open(landwehr_text_output_path, 'w') as f:
            f.write("\n\n".join(landwehr_facts))

        # 6. Compute statistics over landwehr's output
        landwehr_stats = {
            'num_facts': len(landwehr_facts),
            "num_triples": "Compute this, human !",
            "true_positives": "Compute this, human !",
            "false_positives": "Compute this, human !",
            "true_negatives": "Compute this, human !",
            "precision": "Compute this, human !",
            "recall": "Compute this, human !"
        }

        # 7. Look for information in dbpedia about the topic
        # Basically look if a dbpedia link was defined in the medatada
        # of this topic
        dbpedia_resource = metadata['dbpedia']
        dbpedia_stats = None

        if dbpedia_resource != 'None':
            # 8. Compute statistics from the dbpedia knowledge graph about the topic
            dbpedia_stats = get_statistics(dbpedia_resource)

            # Save the dbpedia resource graph to file in turtle format
            dbpedia_graph = Graph().parse(dbpedia_resource)
            dbpedia_graph.serialize(
                destination=dbpedia_resource_file_path,
                format='turtle')

        # 9. Write statistics to file
        stats = {
            'semantic': semantic_stats,
            'landwehr': landwehr_stats,
            'dbpedia': dbpedia_stats
        }

        with open(stats_file_path, 'w') as f:
            f.write(json.dumps(stats, indent=4))


def read_first_text_file_in_dir(directory):
    """Read the text file in the given directory which has the same name
    as the directory"""
    # Get a list of all files in the directory
    files = os.listdir(directory)
    file_name = directory.split('/')[-1] + '.txt'
    file_path = os.path.join(directory, file_name)

    if file_name in files:
        # Open the text file and read its contents
        with open(file_path, 'r') as file:
            text = file.read()

        return text

    # If no text file was found, raise a FileNotFoundError
    raise FileNotFoundError(f"No text file found in {directory}")


def get_metadata(directory: str):
    metadata_path = os.path.join(directory, 'meta.json')

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"{metadata_path} couldn't be found")

    with open(metadata_path, "r") as f:
        text = f.read()
        metadata_json = json.loads(text)

    return metadata_json


def get_statistics(kg_path: str) -> dict:
    """Compute general statistics about a given knowledge graph"""
    # Create an RDF graph
    graph = Graph()

    # Load RDF data from the provided kg_path into the graph
    graph.parse(kg_path)

    # Count the number of triples in the graph
    num_triples = len(graph)

    # Return the number of triples
    return {
        "num_triples": num_triples,
        "true_positives": "Compute this, human !",
        "false_positives": "Compute this, human !",
        "true_negatives": "Compute this, human !",
        "precision": "Compute this, human !",
        "recall": "Compute this, human !"
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Validation pipeline for knowledge graph extraction")
    parser.add_argument("-t", "--topics",
                        nargs="+",
                        help="Specify folders inside validation/topics to process")
    args = parser.parse_args()
    main(args)
