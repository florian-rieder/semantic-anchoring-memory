# Semantic Anchoring for Long-Term Conversational Memory

## Goal
The goal of this project is to give a long-term memory to a conversational LLM. The aim is to reproduce the architecture outlined in Landwehr et al. (2023) [[1](#references)] to use as a baseline, and to create a new memory creation pipeline based on the storage of atomic facts in the form of semantic triplets, which could allow for a memory which captures the rich relationships between concepts and entities.



## Task list

### General
- [x] Chat with the user in the command line

### Reproducing Landwehr et al.
- [x] Extract facts from conversation
- [ ] Store facts as memories in a persistent way
- [ ] Recall relevant stored facts during a conversation
- [ ] Generate responses using these facts

### Semantic Memory
- [x] Create a custom Memory module
- [ ] Extract triplets from conversation
- [ ] Store triplets in knowledge graph
- [ ] Store triplets in vector store
- [ ] Recall triplets based on a similarity search
- [ ] Recall triplets using Named Entity Recognition to do a graph search


## References
[[1](#goal)] Landwehr, Fabian, Erika Varis Doggett, and Romann M. Weber (Sept. 2023). “Memories for Virtual AI Characters”. In: *Proceedings of the 16th International Natural Language Generation Conference*. INLG-SIGDIAL 2023. Ed. by C. Maria Keet, Hung-Yi Lee, and Sina Zarrieß. Prague, Czechia: Association for Computational Linguistics, pp. 237–252. doi: 10.18653/v1/2023.inlg-main.17. url: [https://aclanthology.org/2023.inlg-main.17](https://aclanthology.org/2023.inlg-main.17) (visited on 11/28/2023)