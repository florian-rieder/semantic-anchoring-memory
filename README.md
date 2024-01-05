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
- [ ] Encode triplets into RDF
- [ ] Store triplets in knowledge graph
- [ ] Recall entities based on a similarity search
- [ ] Recall triplets using Named Entity Recognition to do a graph search


## Memory architecture
![Memory architecture diagram](https://github.com/florian-rieder/semantic-anchoring-memory/assets/48287183/6b4d7ad1-5bbb-4457-bf50-16debba4e77d)


## Installation

```bash
python -m venv venv
```

```bash
. venv/bin/activate
```

```bash
pip install -r requirements.txt
```

### Generate classes and predicates vector stores

```bash
python tbox.py
```

### Launch the chat in the command line
```bash
python cli.py
```

### Launch the chat with the web interface
```bash
uvicorn main:app
```

## References
[[1](#goal)] Landwehr, Fabian, Erika Varis Doggett, and Romann M. Weber (Sept. 2023). “Memories for Virtual AI Characters”. In: *Proceedings of the 16th International Natural Language Generation Conference*. INLG-SIGDIAL 2023. Ed. by C. Maria Keet, Hung-Yi Lee, and Sina Zarrieß. Prague, Czechia: Association for Computational Linguistics, pp. 237–252. doi: 10.18653/v1/2023.inlg-main.17. url: [https://aclanthology.org/2023.inlg-main.17](https://aclanthology.org/2023.inlg-main.17) (visited on 11/28/2023)