"""Quick, hard-coded Azure SQL fetch using Azure CLI auth and pyodbc.

Run with: `python -m agent.azure_sql_hardcoded`
Requires `az login` to have been completed in the shell.
"""

from __future__ import annotations

import struct

import pyodbc
from azure.identity import AzureCliCredential

# Hard-coded configuration for now.
SERVER = "prod-platform-wellfit-sqlserver-reporting.database.windows.net"
DATABASE = "analytics-dev"
SCOPE = "https://database.windows.net/.default"
QUERY = "SELECT TOP 10 * FROM [dbo].[DimDate]"

# SQL_COPT_SS_ACCESS_TOKEN constant for pyodbc attrs_before.
SQL_COPT_SS_ACCESS_TOKEN = 1256


def get_connection() -> pyodbc.Connection:
    """Return a pyodbc connection that injects an Azure AD token."""
    credential = AzureCliCredential()
    token = credential.get_token(SCOPE)
    token_bytes = token.token.encode("utf-16-le")
    token_struct = struct.pack(f"<I{len(token_bytes)}s", len(token_bytes), token_bytes)

    connection_string = (
        "Driver={ODBC Driver 17 for SQL Server};"
        f"Server=tcp:{SERVER},1433;"
        f"Database={DATABASE};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )
    return pyodbc.connect(connection_string, attrs_before={SQL_COPT_SS_ACCESS_TOKEN: token_struct})


def main() -> None:
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(QUERY)
                row = cursor.fetchone()
                if row:
                    print(f"Connection Successful! First row: {row}")
                else:
                    print("Connection succeeded but no rows returned.")
    except pyodbc.Error as ex:
        sqlstate = ex.args[0] if ex.args else "UNKNOWN"
        print(f"Connection Error: {sqlstate} - {ex}")


if __name__ == "__main__":
    main()
