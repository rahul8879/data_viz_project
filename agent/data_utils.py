"""Utility functions for configuring SQL connections."""

from __future__ import annotations

import struct
import urllib.parse
from pathlib import Path
from typing import Dict

from azure.identity import AzureCliCredential

# Constant used by pyodbc to pass an access token through attrs_before.
SQL_COPT_SS_ACCESS_TOKEN = 1256
AZURE_SQL_SCOPE = "https://database.windows.net/.default"


def validate_table_name(table_name: str) -> None:
    """Guard against SQL injection inside table names."""
    # Allow optional schema prefix (e.g., dbo.table or rpt.my_table)
    allowed = table_name.replace("_", "").replace(".", "")
    if not allowed.isalnum() or table_name.count(".") > 1:
        raise ValueError("Table names may only contain letters, numbers, underscores, and a single schema prefix.")


def build_sqlite_uri(db_path: Path) -> str:
    """Return a SQLAlchemy SQLite URI for the provided path."""
    return f"sqlite:///{db_path}"


def is_sqlite_uri(db_uri: str) -> bool:
    """Detect whether the provided URI targets SQLite."""
    return db_uri.startswith("sqlite")


def build_azure_sqlalchemy_uri(
    server: str,
    database: str,
    driver: str = "ODBC Driver 17 for SQL Server",
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Construct an encoded SQLAlchemy URI for Azure SQL over pyodbc."""
    cred = ""
    if username and password:
        cred = f"Uid={username};Pwd={password};"
    odbc_str = (
        f"Driver={{{driver}}};"
        f"Server=tcp:{server},1433;"
        f"Database={database};"
        f"{cred}"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    encoded = urllib.parse.quote_plus(odbc_str)
    return f"mssql+pyodbc:///?odbc_connect={encoded}"


def build_azure_cli_access_token_args(scope: str = AZURE_SQL_SCOPE) -> Dict[str, Dict[int, bytes]]:
    """Return connect_args that inject a short-lived Azure AD access token via Azure CLI auth."""
    credential = AzureCliCredential()
    token = credential.get_token(scope)
    token_bytes = token.token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)
    return {"attrs_before": {SQL_COPT_SS_ACCESS_TOKEN: token_struct}}
