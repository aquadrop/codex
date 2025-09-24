"""Document ingestion utilities for GraphRAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .graph_service import GraphService

try:  # pragma: no cover - optional dependency
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback implementation

    class RecursiveCharacterTextSplitter:  # type: ignore[override]
        """Minimal fallback splitter compatible with LangChain's interface."""

        def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, text: str) -> List[str]:
            if not text:
                return []
            chunks: List[str] = []
            start = 0
            while start < len(text):
                end = min(len(text), start + self.chunk_size)
                chunks.append(text[start:end])
                start = end - self.chunk_overlap
                if start < 0:
                    start = 0
            return chunks


@dataclass
class DocumentIngestor:
    """Convert raw documents into graph nodes and relationships."""

    graph_service: GraphService
    chunk_size: int = 500
    chunk_overlap: int = 50
    splitter: RecursiveCharacterTextSplitter | None = field(default=None)

    def __post_init__(self) -> None:
        if self.splitter is None:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

    def ingest(
        self,
        document_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a document into the graph database.

        The method creates a ``Document`` node with ``DocumentChunk`` children,
        linking them sequentially to preserve ordering.
        """

        if not document_id:
            raise ValueError("document_id is required")

        metadata = metadata or {}
        document_properties = {"document_id": document_id, **metadata}
        document_node = self.graph_service.create_node("Document", document_properties)

        chunk_nodes: List[Any] = []
        chunks = self.splitter.split_text(text)
        for index, chunk in enumerate(chunks):
            chunk_properties = {
                "document_id": document_id,
                "chunk_index": index,
                "text": chunk,
            }
            chunk_node = self.graph_service.create_node("DocumentChunk", chunk_properties)
            chunk_nodes.append(chunk_node)
            self.graph_service.create_relationship(
                "Document",
                {"document_id": document_id},
                "DocumentChunk",
                {"document_id": document_id, "chunk_index": index},
                "HAS_CHUNK",
                {"order": index},
            )
            if index > 0:
                self.graph_service.create_relationship(
                    "DocumentChunk",
                    {"document_id": document_id, "chunk_index": index - 1},
                    "DocumentChunk",
                    {"document_id": document_id, "chunk_index": index},
                    "NEXT_CHUNK",
                    {"order": index},
                )

        return {
            "document": document_node,
            "chunks": chunk_nodes,
        }


__all__ = ["DocumentIngestor"]
