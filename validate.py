"""
Validation pipeline for the knowledge graph extraction
"""

import logging
import os
import shutil
from tqdm import tqdm

from rdflib import Graph

from ingest import init
from memory.semantic.learn import memorize


logger = logging.getLogger(__name__)


def main():

    # Define the directory containing the subject directories
    subjects_dir = f'{os.getcwd()}/validation/subjects/'

    # Get a list of all directories in the subjects directory
    subject_directories = sorted(os.listdir(subjects_dir))

    print(subject_directories)

    # Loop through each subject directory
    for subject_directory in tqdm(subject_directories):
        print(subject_directory)
    

        # Construct full paths
        subject_dir_path = os.path.join(subjects_dir, subject_directory)
        output_kg_path = os.path.join(subject_dir_path, 'output.ttl')
        entities_db_path = os.path.join(subject_dir_path, 'entities_db')

        # Delete knowledge graph and/or entities db if they already exist
        if os.path.exists(output_kg_path):
            os.remove(output_kg_path)
        if os.path.exists(entities_db_path):
            shutil.rmtree(entities_db_path)

        # 0. Initialize memory
        llm, store = init(
            # Save the output graph in the subject directory
            memory_path=output_kg_path,
            # Save the entities db associated to the graph in the subject directory
            entities_db_path=entities_db_path,
            # Use no prior knowledge
            base_knowledge=None
        )

        # 1. Get the text file
        # Check if the path is a directory
        if not os.path.isdir(subject_dir_path):
            logger.warning(f'Not a valid directory {subject_dir_path}')
            continue

        # Call the function to read text files within the subject directory
        text = read_text_file(subject_dir_path)

        # 2. Process the text file and output a knowledge graph
        memorize(text, llm, store)

        # 3. Compute statistics over the resulting knowledge graph
        stats = get_statistics(output_kg_path)
        print(stats)

        # 4. Look for information in dbpedia about the subject

        # 5. Compute statistics from the dbpedia knowledge graph about the subject


def read_text_file(directory):
    """Read the first text file in the given directory"""
    # Get a list of all files in the directory
    files = os.listdir(directory)

    # Loop through each file in the directory
    for file_name in files:
        # Check if the file is a text file
        if not file_name.endswith('.txt'):
            continue

        # Construct the full path to the text file
        file_path = os.path.join(directory, file_name)

        # Open the text file and read its contents
        with open(file_path, 'r') as file:
            text = file.read()

        return text


def get_statistics(kg_path: str):
    # Create an RDF graph
    graph = Graph()

    # Load RDF data from the provided kg_path into the graph
    graph.parse(kg_path)

    # Count the number of triples in the graph
    num_triples = len(graph)

    # Return the number of triples
    return num_triples


if __name__ == '__main__':
    main()