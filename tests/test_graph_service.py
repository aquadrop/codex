from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from graphrag.config import Neo4jConfig
from graphrag.graph_service import GraphService


@pytest.fixture
def mock_driver(monkeypatch):
    driver = MagicMock()
    session = MagicMock()
    driver.session.return_value.__enter__.return_value = session
    monkeypatch.setattr("graphrag.graph_service.GraphDatabase.driver", lambda *_, **__: driver)
    return driver, session


def make_record(**data):
    record = MagicMock()
    record.data.return_value = data
    return record


def test_create_node_executes_cypher(mock_driver):
    driver, session = mock_driver
    session.run.return_value = [make_record(n={"name": "Alice"})]
    service = GraphService(Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass"))

    node = service.create_node("Person", {"name": "Alice"})

    session.run.assert_called_once()
    query, params = session.run.call_args[0][0], session.run.call_args[0][1]
    assert query.startswith("CREATE (n:Person)")
    assert params["props"] == {"name": "Alice"}
    assert node == {"name": "Alice"}


def test_get_nodes_with_filter_and_limit(mock_driver):
    driver, session = mock_driver
    session.run.return_value = [make_record(n={"name": "Alice"}), make_record(n={"name": "Bob"})]
    service = GraphService(Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass"))

    nodes = service.get_nodes("Person", {"name": "Alice"}, limit=1)

    session.run.assert_called_once()
    assert "LIMIT $limit" in session.run.call_args[0][0]
    assert session.run.call_args[0][1]["limit"] == 1
    assert nodes == [{"name": "Alice"}, {"name": "Bob"}]


def test_update_node_returns_count(mock_driver):
    driver, session = mock_driver
    session.run.return_value = [make_record(count=2)]
    service = GraphService(Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass"))

    count = service.update_node("Person", {"name": "Alice"}, {"age": 30})

    assert count == 2
    query = session.run.call_args[0][0]
    assert query.startswith("MATCH (n:Person")
    assert "SET n += $updates" in query


def test_delete_relationship(mock_driver):
    driver, session = mock_driver
    session.run.return_value = [make_record(count=1)]
    service = GraphService(Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass"))

    deleted = service.delete_relationship(
        "Person", {"name": "Alice"}, "Project", {"name": "Apollo"}, "WORKS_ON"
    )

    assert deleted == 1
    query = session.run.call_args[0][0]
    assert query.startswith("MATCH (from_node:Person")
    assert "DELETE r" in query
