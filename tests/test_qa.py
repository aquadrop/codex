from __future__ import annotations

from dataclasses import dataclass

import pytest

from graphrag.config import Neo4jConfig
from graphrag.qa import GraphQuestionAnswerer


@dataclass
class DummyChain:
    response: dict

    def invoke(self, payload):
        return self.response | {"payload": payload}


def test_answer_uses_provided_chain():
    config = Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass")
    chain = DummyChain({"result": "42"})
    qa = GraphQuestionAnswerer(config=config, chain=chain)

    answer = qa.answer("life?")

    assert answer["result"] == "42"
    assert answer["payload"]["query"] == "life?"


def test_generate_cypher_extracts_query():
    class SpyChain:
        def invoke(self, payload):
            return {
                "intermediate_steps": [
                    {"cypher": "MATCH (n) RETURN n"},
                ]
            }

    config = Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass")
    qa = GraphQuestionAnswerer(config=config, chain=SpyChain())

    cypher = qa.generate_cypher("life?")

    assert cypher == "MATCH (n) RETURN n"


def test_answer_requires_question():
    config = Neo4jConfig(uri="bolt://localhost", username="neo4j", password="pass")
    qa = GraphQuestionAnswerer(config=config, chain=DummyChain({}))

    with pytest.raises(ValueError):
        qa.answer("")
