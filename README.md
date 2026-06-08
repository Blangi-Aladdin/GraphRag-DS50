# GraphRAG-DS50

Interactive GraphRAG system with Human-in-the-Loop knowledge graph refinement.

## Project Overview

This project implements a complete GraphRAG pipeline that:

- Extracts knowledge graph triplets from documents using LLMs
- Stores knowledge in Neo4j
- Visualizes the graph interactively
- Supports Human-in-the-Loop graph refinement
- Answers questions using GraphRAG retrieval
- Evaluates generated answers using ROUGE and BERTScore

## Features

### Document Processing

- TXT support
- PDF support
- DOCX support
- JSON support
- Configurable chunking

### Knowledge Graph Construction

- Automatic triplet extraction
- Neo4j integration
- Graph normalization
- Graph visualization

### Human-in-the-Loop

- Edit extracted triplets
- Add triplets manually
- Delete nodes
- Delete relationships
- Rename nodes
- Merge nodes
- Add relationships
- Edit relationship types

### GraphRAG Chatbot

- Multi-hop retrieval
- Neo4j-based reasoning
- Chat history

### Evaluation

- ROUGE-1
- ROUGE-2
- ROUGE-L
- BERTScore F1

## Technology Stack

- Python
- Streamlit
- OpenAI
- Neo4j
- PyVis
- LangChain Text Splitters

## Installation

```bash
pip install -r requirements.txt
```

## Required Environment Variables

```text
OPENAI_API_KEY
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
NEO4J_DATABASE
```

## Authors

UTBM – DS50 Project

Ala Eddine Blangi
