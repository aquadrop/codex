from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag.ingest import DocumentIngestor


def test_ingest_creates_document_and_chunks():
    graph_service = MagicMock()
    graph_service.create_node.side_effect = [
        {"document_id": "doc-1"},
        {"chunk_index": 0},
        {"chunk_index": 1},
    ]

    ingestor = DocumentIngestor(graph_service, chunk_size=10, chunk_overlap=0)
    result = ingestor.ingest("doc-1", "hello world", metadata={"source": "test"})

    assert result["document"] == {"document_id": "doc-1"}
    assert len(result["chunks"]) == 2
    graph_service.create_relationship.assert_any_call(
        "Document",
        {"document_id": "doc-1"},
        "DocumentChunk",
        {"document_id": "doc-1", "chunk_index": 0},
        "HAS_CHUNK",
        {"order": 0},
    )
    graph_service.create_relationship.assert_any_call(
        "DocumentChunk",
        {"document_id": "doc-1", "chunk_index": 0},
        "DocumentChunk",
        {"document_id": "doc-1", "chunk_index": 1},
        "NEXT_CHUNK",
        {"order": 1},
    )


def test_ingest_requires_document_id():
    graph_service = MagicMock()
    ingestor = DocumentIngestor(graph_service)

    with pytest.raises(ValueError):
        ingestor.ingest("", "text")
