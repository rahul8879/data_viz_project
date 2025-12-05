"""Utility functions for configuring SQL connections."""

from __future__ import annotations

from pathlib import Path



def validate_table_name(table_name: str) -> None:
    """Guard against SQL injection inside table names."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Table names may only contain letters, numbers, and underscores.")


def build_sqlite_uri(db_path: Path) -> str:
    """Return a SQLAlchemy SQLite URI for the provided path."""
    return f"sqlite:///{db_path}"


def is_sqlite_uri(db_uri: str) -> bool:
    """Detect whether the provided URI targets SQLite."""
    return db_uri.startswith("sqlite")
