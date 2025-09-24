"""Configuration helpers for the GraphRAG project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class Neo4jConfig:
    """Configuration required to connect to a Neo4j instance.

    The class provides convenience constructors for loading configuration
    from environment variables which keeps secrets out of source control.
    """

    uri: str
    username: str
    password: str
    database: Optional[str] = None

    @classmethod
    def from_env(
        cls,
        uri_var: str = "NEO4J_URI",
        user_var: str = "NEO4J_USERNAME",
        password_var: str = "NEO4J_PASSWORD",
        database_var: str = "NEO4J_DATABASE",
    ) -> "Neo4jConfig":
        """Create an instance from environment variables.

        Args:
            uri_var: Environment variable holding the Bolt URL.
            user_var: Environment variable holding the username.
            password_var: Environment variable holding the password.
            database_var: Environment variable with the database name (optional).

        Returns:
            Neo4jConfig: Parsed configuration.

        Raises:
            KeyError: If required environment variables are missing.
        """

        uri = os.environ[uri_var]
        username = os.environ[user_var]
        password = os.environ[password_var]
        database = os.environ.get(database_var)
        return cls(uri=uri, username=username, password=password, database=database)


__all__ = ["Neo4jConfig"]
