# Multi-Hop Contextual Document Retrieval Engine

A lightweight, LLM-free document retrieval system that identifies relationships across multiple documents and returns answers with traceable evidence.

## Features

- Document processing and sentence-level indexing
- Entity and relation extraction
- Semantic retrieval using sentence embeddings
- Multi-hop reasoning over a local knowledge graph
- Evidence linking and source traceability
- Ambiguous-query and insufficient-evidence handling
- Custom evaluation dataset and metrics

## Technology

- Python
- spaCy
- sentence-transformers
- NetworkX
- NumPy

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run

```bash
python scripts/build_index.py
python scripts/run_query.py "Which product was launched by the company acquired by the organization that employed Employee X?"
python scripts/evaluate.py
```

## Architecture

Documents are split into passages. Entities and relations are extracted from each passage and stored in a directed knowledge graph. Queries are matched semantically against passages and resolved through multi-hop graph paths.

The system returns:

- Answer
- Confidence score
- Relationship chain
- Original supporting passages

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

The evaluation measures answer accuracy, evidence recall, evidence precision, mean reciprocal rank, and latency.
