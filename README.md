# GraphRAG with LangChain and Neo4j

This project provides a minimal but production-ready scaffold for building a Graph-based Retrieval Augmented Generation (GraphRAG) system on top of [LangChain](https://python.langchain.com) and [Neo4j](https://neo4j.com/).

The package exposes utilities for:

- Managing Neo4j connections and executing CRUD operations.
- Ingesting raw documents into a graph structure suitable for retrieval.
- Running multi-hop question answering over the graph using LangChain's `GraphCypherQAChain`.
- A command line interface (CLI) that wires these components together for easy experimentation.

## Getting started

### Installation

```bash
pip install -e .[dev]
```

The QA components require access to an LLM. The CLI is configured for OpenAI's Chat models via `langchain-openai`, but any LangChain-compatible LLM can be supplied programmatically.

### Environment variables

Set the following environment variables to connect to your Neo4j instance:

- `NEO4J_URI` – bolt URI (e.g. `bolt://localhost:7687`)
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (optional)

For the QA command you will also need `OPENAI_API_KEY` (or pass `--openai-api-key`).

### CLI usage

Create a node:

```bash
python -m graphrag.cli create-node Person --properties name="Alice" age=28
```

Ingest a document:

```bash
python -m graphrag.cli ingest doc-001 ./path/to/file.txt --metadata source="manual"
```

Ask a multi-hop question:

```bash
python -m graphrag.cli ask "How is Alice connected to the marketing project?"
```

### Python API

```python
from graphrag import Neo4jConfig, GraphService, DocumentIngestor, GraphQuestionAnswerer
from langchain_openai import ChatOpenAI

config = Neo4jConfig.from_env()
service = GraphService(config)
ingestor = DocumentIngestor(service)
answerer = GraphQuestionAnswerer(config=config, llm=ChatOpenAI())
```

Refer to the module docstrings for more details.

## Testing

```bash
pytest
```

The tests use mocks and do not require a running Neo4j instance.
