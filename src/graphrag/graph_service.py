"""Utilities for interacting with a Neo4j database."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover - allow tests without driver

    class GraphDatabase:  # type: ignore[override]
        @staticmethod
        def driver(*args, **kwargs):
            raise ImportError("The neo4j Python driver is not installed")

from .config import Neo4jConfig


@dataclass
class GraphService:
    """High level CRUD helper around the Neo4j Python driver."""

    config: Neo4jConfig
    driver_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._driver = GraphDatabase.driver(
            self.config.uri,
            auth=(self.config.username, self.config.password),
            **self.driver_kwargs,
        )

    def close(self) -> None:
        """Close the underlying driver."""

        if self._driver:
            self._driver.close()

    # ---------------------------- CRUD operations -------------------------

    def create_node(self, label: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Create a node with the given label and properties."""

        query = f"CREATE (n:{label}) SET n = $props RETURN n"
        result = self._run(query, {"props": properties})
        return result[0]["n"] if result else {}

    def get_nodes(
        self,
        label: str,
        match_properties: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve nodes optionally filtered by properties."""

        conditions, params = self._build_match("n", match_properties)
        query = f"MATCH (n:{label}{conditions}) RETURN n"
        if limit:
            query += " LIMIT $limit"
            params["limit"] = limit
        result = self._run(query, params)
        return [record["n"] for record in result]

    def update_node(
        self,
        label: str,
        match_properties: Dict[str, Any],
        updates: Dict[str, Any],
    ) -> int:
        """Update properties on matching nodes.

        Returns:
            Number of nodes updated.
        """

        conditions, params = self._build_match("n", match_properties)
        params["updates"] = updates
        query = f"MATCH (n:{label}{conditions}) SET n += $updates RETURN count(n) as count"
        result = self._run(query, params)
        return int(result[0]["count"]) if result else 0

    def delete_node(self, label: str, match_properties: Dict[str, Any]) -> int:
        """Delete nodes matching the provided properties."""

        conditions, params = self._build_match("n", match_properties)
        query = f"MATCH (n:{label}{conditions}) DETACH DELETE n RETURN count(n) as count"
        result = self._run(query, params)
        return int(result[0]["count"]) if result else 0

    def create_relationship(
        self,
        from_label: str,
        from_match: Dict[str, Any],
        to_label: str,
        to_match: Dict[str, Any],
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a relationship between two nodes."""

        from_conditions, from_params = self._build_match("from", from_match)
        to_conditions, to_params = self._build_match("to", to_match)
        params = {**from_params, **to_params}
        params["props"] = properties or {}
        query = (
            f"MATCH (from_node:{from_label}{from_conditions}), (to_node:{to_label}{to_conditions}) "
            f"CREATE (from_node)-[r:{rel_type}]->(to_node) SET r += $props RETURN r"
        )
        result = self._run(query, params)
        return result[0]["r"] if result else {}

    def delete_relationship(
        self,
        from_label: str,
        from_match: Dict[str, Any],
        to_label: str,
        to_match: Dict[str, Any],
        rel_type: str,
    ) -> int:
        """Delete relationships between matching nodes."""

        from_conditions, from_params = self._build_match("from", from_match)
        to_conditions, to_params = self._build_match("to", to_match)
        params = {**from_params, **to_params}
        query = (
            f"MATCH (from_node:{from_label}{from_conditions})-[r:{rel_type}]->(to_node:{to_label}{to_conditions}) "
            "DELETE r RETURN count(r) as count"
        )
        result = self._run(query, params)
        return int(result[0]["count"]) if result else 0

    # ---------------------------- Helper methods -------------------------

    def _run(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a Cypher query against the configured database."""

        with self._driver.session(database=self.config.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    @staticmethod
    def _build_match(alias: str, match_properties: Optional[Dict[str, Any]]) -> tuple[str, Dict[str, Any]]:
        if not match_properties:
            return "", {}
        conditions: List[str] = []
        params: Dict[str, Any] = {}
        for index, (key, value) in enumerate(match_properties.items()):
            param_name = f"{alias}_{index}"
            conditions.append(f"{key}: ${param_name}")
            params[param_name] = value
        joined = ", ".join(conditions)
        return (f" {{{joined}}}", params) if conditions else ("", params)


__all__ = ["GraphService"]
