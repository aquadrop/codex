"""Command line interface for managing the GraphRAG project."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import typer

from .config import Neo4jConfig
from .graph_service import GraphService
from .ingest import DocumentIngestor
from .qa import GraphQuestionAnswerer

try:  # pragma: no cover - optional dependency
    from langchain_openai import ChatOpenAI
except Exception:  # pragma: no cover
    ChatOpenAI = None  # type: ignore[assignment]

app = typer.Typer(help="Utilities for interacting with the GraphRAG stack.")


def _parse_key_value_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Invalid property '{pair}'. Use key=value format.")
        key, raw_value = pair.split("=", 1)
        try:
            parsed[key] = json.loads(raw_value)
        except json.JSONDecodeError:
            parsed[key] = raw_value
    return parsed


@contextmanager
def _graph_service(config: Neo4jConfig) -> GraphService:
    service = GraphService(config)
    try:
        yield service
    finally:
        service.close()


def _load_config(
    uri: Optional[str], username: Optional[str], password: Optional[str], database: Optional[str]
) -> Neo4jConfig:
    if uri and username and password:
        return Neo4jConfig(uri=uri, username=username, password=password, database=database)
    return Neo4jConfig.from_env()


@app.command("create-node")
def create_node(
    label: str,
    properties: Optional[list[str]] = typer.Option(None, help="key=value properties for the node"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    username: Optional[str] = typer.Option(None, envvar="NEO4J_USERNAME"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    database: Optional[str] = typer.Option(None, envvar="NEO4J_DATABASE"),
) -> None:
    """Create a node in the graph."""

    config = _load_config(uri, username, password, database)
    props = _parse_key_value_pairs(properties or [])
    with _graph_service(config) as service:
        node = service.create_node(label, props)
        typer.echo(json.dumps(node, default=str, indent=2))


@app.command("get-nodes")
def get_nodes(
    label: str,
    filters: Optional[list[str]] = typer.Option(None, help="key=value filters"),
    limit: Optional[int] = typer.Option(None, help="Maximum number of nodes to return"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    username: Optional[str] = typer.Option(None, envvar="NEO4J_USERNAME"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    database: Optional[str] = typer.Option(None, envvar="NEO4J_DATABASE"),
) -> None:
    """Retrieve nodes from the graph."""

    config = _load_config(uri, username, password, database)
    params = _parse_key_value_pairs(filters or [])
    with _graph_service(config) as service:
        nodes = service.get_nodes(label, params or None, limit=limit)
        typer.echo(json.dumps(nodes, default=str, indent=2))


@app.command("update-node")
def update_node(
    label: str,
    match: list[str] = typer.Option(..., help="key=value matcher"),
    updates: list[str] = typer.Option(..., help="key=value updates"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    username: Optional[str] = typer.Option(None, envvar="NEO4J_USERNAME"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    database: Optional[str] = typer.Option(None, envvar="NEO4J_DATABASE"),
) -> None:
    """Update properties on nodes."""

    config = _load_config(uri, username, password, database)
    match_props = _parse_key_value_pairs(match)
    update_props = _parse_key_value_pairs(updates)
    with _graph_service(config) as service:
        count = service.update_node(label, match_props, update_props)
        typer.echo(json.dumps({"updated": count}))


@app.command("delete-node")
def delete_node(
    label: str,
    match: list[str] = typer.Option(..., help="key=value matcher"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    username: Optional[str] = typer.Option(None, envvar="NEO4J_USERNAME"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    database: Optional[str] = typer.Option(None, envvar="NEO4J_DATABASE"),
) -> None:
    """Delete matching nodes."""

    config = _load_config(uri, username, password, database)
    match_props = _parse_key_value_pairs(match)
    with _graph_service(config) as service:
        count = service.delete_node(label, match_props)
        typer.echo(json.dumps({"deleted": count}))


@app.command("ingest")
def ingest(
    document_id: str,
    path: Path = typer.Argument(..., exists=True, readable=True),
    metadata: Optional[list[str]] = typer.Option(None, help="key=value metadata"),
    chunk_size: int = typer.Option(500, help="Document chunk size"),
    chunk_overlap: int = typer.Option(50, help="Document chunk overlap"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    username: Optional[str] = typer.Option(None, envvar="NEO4J_USERNAME"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    database: Optional[str] = typer.Option(None, envvar="NEO4J_DATABASE"),
) -> None:
    """Ingest a text document into the graph."""

    config = _load_config(uri, username, password, database)
    meta_props = _parse_key_value_pairs(metadata or [])
    text = path.read_text(encoding="utf-8")
    with _graph_service(config) as service:
        ingestor = DocumentIngestor(service, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        result = ingestor.ingest(document_id=document_id, text=text, metadata=meta_props)
        typer.echo(json.dumps(result, default=str, indent=2))


@app.command("ask")
def ask(
    question: str,
    model: str = typer.Option("gpt-3.5-turbo", help="Model name for the OpenAI Chat API"),
    openai_api_key: Optional[str] = typer.Option(None, envvar="OPENAI_API_KEY"),
    uri: Optional[str] = typer.Option(None, envvar="NEO4J_URI"),
    username: Optional[str] = typer.Option(None, envvar="NEO4J_USERNAME"),
    password: Optional[str] = typer.Option(None, envvar="NEO4J_PASSWORD"),
    database: Optional[str] = typer.Option(None, envvar="NEO4J_DATABASE"),
) -> None:
    """Ask a multi-hop question over the knowledge graph."""

    if ChatOpenAI is None:
        raise typer.BadParameter("langchain-openai is required for question answering")
    config = _load_config(uri, username, password, database)
    if not openai_api_key:
        raise typer.BadParameter("An OpenAI API key must be provided via --openai-api-key or environment")
    llm = ChatOpenAI(model=model, openai_api_key=openai_api_key)
    answerer = GraphQuestionAnswerer(config=config, llm=llm)
    result = answerer.answer(question)
    typer.echo(json.dumps(result, default=str, indent=2))


def main() -> None:  # pragma: no cover - entry point wrapper
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
