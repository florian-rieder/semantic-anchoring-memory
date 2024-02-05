# Semantic Anchoring for Long-Term Conversational Memory


## Goal
The goal of this project is to give a long-term memory to a conversational LLM. The aim is to reproduce the architecture outlined in Landwehr et al. (2023) [[1](#references)] to use as a baseline, and to create a new memory creation pipeline based on the storage of atomic facts in the form of semantic triplets, which could allow for a memory which captures the rich relationships between concepts and entities.


## Task list

### General
- [x] Chat with the user in the command line
- [x] Chat in a web interface

### Reproducing Landwehr et al.
- [x] Extract facts from texts
- [x] Store facts as memories in a persistent vector store
- [x] Recall relevant stored facts during a conversation
- [x] Generate responses using these facts

### Semantic Memory
- [x] Create a custom Memory module
- [x] Extract triplets from conversation
- [x] Encode triplets into RDF
- [x] Store triplets in knowledge graph
- [x] Recall entities based on a similarity search
- [x] Recall triplets using Named Entity Recognition to do a graph search


## Installation

```bash
python -m venv venv
```

```bash
. venv/bin/activate
```

```bash
# Install poetry
pip install poetry

# Use poetry to resolve dependencies
poetry install --no-root
```

### Docker build

```bash
docker build . -t myimage

docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_API_KEY myimage
```

### Configuration
Start by configuring which ontologies T-Boxes should be used as a world model, by editing the `ONTOLOGIES_PATHS` list to include any T-Box you might want. Local files paths (in the `ontologies` directory) or http addresses of networked resources can be used.

Add your OpenAI API key to the environment variables using:
```bash
export OPENAI_API_KEY=my-open-ai-key
```

### Generate classes and predicates vector stores
First off, you will need to populate the classes and predicates vector stores with the classes and predicates from your chosen ontologies T-Boxes.
```bash
python generate_tbox_db.py
```

### Ingest arbitrary files

```bash
python ingest.py -f file1.txt file2.txt
```

### Launch the chat in the command line
```bash
python cli.py

# You can specify which memory module to use:
python cli.py --memory landwehr
# or
python cli.py -m semantic
```

### Launch the chat with the web interface
```bash
uvicorn main:app
```


## Project structure

- `config.py`: Configuration file, containing paths to databases and notably the list of ontologies to use as memory world model.
- `chat.py`: The main conversation chain, which uses memory modules from the `memory` directory.
- `cli.py`: Command-line interface for chatting with the model through the terminal.
- `main.py`: Web interface for chatting with the model more comfortably. For testing purposes.
- `generate_tbox_db.py`: Script used to precompute the T-Box vector databases. Needs to be run once after having configured the desired ontologies in `config.py`.
- `ingest.py`: Script used to ingest an arbitrary text into the semantic memory.

- `memory`: This directory contains code related to the memory logic of both the Landwehr et al. inspired memory, and the semantic memory.
    - `landwehr`: Landwehr memory module
        - `landwehr.py`: Contains the Langchain memory module and associated functions used to replicate the memory architecture outlined in their paper.
    - `semantic`: Semantic memory module
        - `abox.py`: ABox class, used to define the entity knowledge, which is essentially the long-term memory.
        - `tbox.py`: TBox class, used to retrieve knowledge about the world model to use for the memory
        - `memory.py`: Langchain memory module
        - `store.py`: Interface used by the memory module to access, retrieve and update the memory
        - `learn.py`: Collection of functions used to ingest a raw text and memorize it. Used by `memory.py`
    - `forget.py`: Forgetting curve function, outlined in Landwehr et al., but remained unused in this project for time constraints.
    - `prompts.py`: Prompts used by memory systems.
- `database`: Directory containing T-Box vector databases and long term memories
- `server`: Web interface directory
    - `static`: Static files
        - `css`
            - `styles.css`: Styles of the client interface
        - `js`
            - `main.js`: Body of the client application
            - `textarea.js`: Text input script
    - `templates`
        - `index.html`: Client interface
    - `callbacks.py`: AsyncCallback used for response streaming
    - `schemas.py`: Chat application internal data exchange formats
- `ontologies`: Ontologies and other RDF data
    - `base_knowledge.ttl`: Turtle file containing the base knowledge used when initiating a new memory graph, for first time use.
    - `classes.owl`: (DEVELOPMENT) List of classes embedding strings obtained from the ontologies while generating the classes vector database
    - `predicates.owl`: (DEVELOPMENT) List of predicates embedding strings obtained from the ontologies while generating the predicates vector database
    - `foaf.owl`: Local copy of the foaf ontology T-Box
    - `dbpedia.owl`: Local copy of the dbpedia ontology T-Box
- `notebooks`: Exploratory analysis



## Diagrams
### Memory architecture
![Memory architecture diagram](https://github.com/florian-rieder/semantic-anchoring-memory/assets/48287183/da1bdce8-6e7a-4716-9850-de72a3d3d5cf)
### Memory creation pipeline
![Memory creation pipeline](https://github.com/florian-rieder/semantic-anchoring-memory/assets/48287183/85e8fc77-131b-4202-9a26-217e6d133be5)


## References
[[1](#goal)] Landwehr, Fabian, Erika Varis Doggett, and Romann M. Weber (Sept. 2023). “Memories for Virtual AI Characters”. In: *Proceedings of the 16th International Natural Language Generation Conference*. INLG-SIGDIAL 2023. Ed. by C. Maria Keet, Hung-Yi Lee, and Sina Zarrieß. Prague, Czechia: Association for Computational Linguistics, pp. 237–252. doi: 10.18653/v1/2023.inlg-main.17. url: [https://aclanthology.org/2023.inlg-main.17](https://aclanthology.org/2023.inlg-main.17) (visited on 11/28/2023)
