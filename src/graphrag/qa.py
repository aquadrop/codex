"""Question answering utilities on top of Neo4j and LangChain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .config import Neo4jConfig

try:  # pragma: no cover - optional dependency
    from langchain.chains import GraphCypherQAChain
    from langchain_community.graphs import Neo4jGraph
except Exception:  # pragma: no cover
    GraphCypherQAChain = None  # type: ignore[assignment]
    Neo4jGraph = None  # type: ignore[assignment]


@dataclass
class GraphQuestionAnswerer:
    """Execute multi-hop question answering over a Neo4j knowledge graph."""

    config: Neo4jConfig
    llm: Optional[Any] = None
    chain: Optional[Any] = None
    graph: Optional[Any] = None
    validate_cypher: bool = True
    top_k: int = 5
    response_key: str = "result"

    def __post_init__(self) -> None:
        if self.chain is None:
            if GraphCypherQAChain is None or Neo4jGraph is None:
                raise ImportError(
                    "LangChain graph components are not installed. Install langchain-community to use QA."
                )
            if self.llm is None:
                raise ValueError("An LLM instance must be provided when chain is not supplied")
            if self.graph is None:
                self.graph = Neo4jGraph(
                    url=self.config.uri,
                    username=self.config.username,
                    password=self.config.password,
                    database=self.config.database,
                )
            self.chain = GraphCypherQAChain.from_llm(
                llm=self.llm,
                graph=self.graph,
                validate_cypher=self.validate_cypher,
                top_k=self.top_k,
            )

    def answer(self, question: str, **kwargs: Any) -> Dict[str, Any]:
        """Run the QA chain and return both answer and generated Cypher."""

        if not question:
            raise ValueError("question is required")
        if self.chain is None:
            raise RuntimeError("QA chain is not initialized")
        response = self.chain.invoke({"query": question, **kwargs})
        if isinstance(response, dict):
            return response
        return {self.response_key: response}

    def generate_cypher(self, question: str) -> str:
        """Return the generated Cypher statement without executing it."""

        if not question:
            raise ValueError("question is required")
        if self.chain is None:
            raise RuntimeError("QA chain is not initialized")
        # GraphCypherQAChain exposes ``cypher`` key in the return dict when ``return_intermediate_steps`` is True.
        result = self.chain.invoke({"query": question, "return_intermediate_steps": True})
        if isinstance(result, dict):
            intermediate = result.get("intermediate_steps", [])
            if intermediate:
                generated = intermediate[0].get("cypher")
                if generated:
                    return generated
        raise RuntimeError("The chain did not return a generated Cypher query")


__all__ = ["GraphQuestionAnswerer"]
