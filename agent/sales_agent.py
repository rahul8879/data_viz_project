from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from operator import add
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .data_utils import (
    build_azure_cli_access_token_args,
    build_azure_sqlalchemy_uri,
    validate_table_name,
    split_schema_table,
)
from sqlalchemy import MetaData

BASE_DIR = Path(__file__).resolve().parents[1]

# Accept either repo-root .env or agent/.env to match IDE setup.
for env_path in (BASE_DIR / ".env", BASE_DIR / "agent" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"Loaded environment variables from {env_path}")

DEFAULT_TABLE = os.getenv("AZURE_SQL_TABLE", "DimDate")
AZURE_SQL_SERVER = os.getenv("AZURE_SQL_SERVER")
AZURE_SQL_DATABASE = os.getenv("AZURE_SQL_DATABASE")
AZURE_SQL_DRIVER = os.getenv("AZURE_SQL_DRIVER", "ODBC Driver 17 for SQL Server")
AZURE_SQL_USERNAME = os.getenv("AZURE_SQL_USERNAME")
AZURE_SQL_PASSWORD = os.getenv("AZURE_SQL_PASSWORD")
AZURE_SQL_USE_CLI_AUTH = os.getenv("AZURE_SQL_USE_CLI_AUTH", "true").lower() == "true"


class AgentState(TypedDict):
    """LangGraph state that tracks the threaded conversation."""

    messages: Annotated[List[BaseMessage], add]


@dataclass
class AgentAnswer:
    content: str


@dataclass
class SalesInsightAgent:
    """LangGraph powered agent that reasons over an Azure SQL table."""

    db_uri: Optional[str] = None
    table_name: str = DEFAULT_TABLE
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    use_azure_cli_auth: bool = AZURE_SQL_USE_CLI_AUTH
    # endpoint = '127.0.0.1:11434'

    def __post_init__(self) -> None:
        validate_table_name(self.table_name)

        schema, base_table = split_schema_table(self.table_name)
        qualified_table = f"{schema}.{base_table}" if schema else base_table

        if not self.db_uri:
            if not (AZURE_SQL_SERVER and AZURE_SQL_DATABASE):
                raise ValueError("Set AZURE_SQL_SERVER and AZURE_SQL_DATABASE in your environment to use Azure SQL.")
            if self.use_azure_cli_auth:
                username = password = None
            else:
                if not (AZURE_SQL_USERNAME and AZURE_SQL_PASSWORD):
                    raise ValueError("Provide AZURE_SQL_USERNAME and AZURE_SQL_PASSWORD or enable AZURE_SQL_USE_CLI_AUTH.")
                username = AZURE_SQL_USERNAME
                password = AZURE_SQL_PASSWORD
            self.db_uri = build_azure_sqlalchemy_uri(
                server=AZURE_SQL_SERVER,
                database=AZURE_SQL_DATABASE,
                driver=AZURE_SQL_DRIVER,
                username=username,
                password=password,
            )

        engine_args = {}
        if self.use_azure_cli_auth:
            engine_args["connect_args"] = build_azure_cli_access_token_args()

        metadata = MetaData(schema=schema) if schema else None
        self.sql_database: SQLDatabase = SQLDatabase.from_uri(
            self.db_uri,
            sample_rows_in_table_info=3,
            include_tables=[base_table],
            metadata=metadata,
            engine_args=engine_args or None,
        )
        schema_snapshot = self.sql_database.get_table_info()
        self.system_message = SystemMessage(
            content=self._build_system_prompt(
                schema=schema_snapshot,
                qualified_table=qualified_table,
            ),
        )
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        sql_tool = QuerySQLDataBaseTool(db=self.sql_database)

        self._llm_with_tools = self.llm.bind_tools([sql_tool])

        graph = StateGraph(AgentState)
        graph.add_node("agent", self._run_agent)
        graph.add_node("tools", ToolNode([sql_tool]))

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        self.graph = graph.compile()

    def _build_system_prompt(self, schema: str, qualified_table: str) -> str:
        return (
            "You are a senior data analyst embedded inside a LangGraph agent. "
            "You have one tool: "
            "`query_sql_db` to run read-only SQL against the database (auto-detect schema, include LIMIT 50 on exploratory queries, never modify data). "
            "Follow these rules:\n"
            "- Start with `query_sql_db` to fetch the minimum data needed. Show the SQL you executed in a fenced code block before summarizing results.\n"
            "- Provide crisp, numeric answers grounded in query results. If the request is impossible with available columns, explain why and suggest the missing data.\n\n"
            f"Only query the table `{qualified_table}`; do not reference or join any other tables.\n"
            f"SQL schema snapshot:\n{schema}\n\n"
        )

    def _run_agent(self, state: AgentState) -> dict:
        response = self._llm_with_tools.invoke([self.system_message, *state["messages"]])
        return {"messages": [response]}

    def ask(self, question: str) -> AgentAnswer:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        state = {"messages": [HumanMessage(content=question)]}
        result = self.graph.invoke(state)
        final_message = result["messages"][-1]
        return AgentAnswer(content=final_message.content)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query an Azure SQL table using a LangGraph-powered agent.",
    )
    parser.add_argument(
        "--db-uri",
        default=None,
        help=(
            "Full SQLAlchemy URI for Azure SQL over pyodbc. "
            "Example: mssql+pyodbc:///?odbc_connect=..."
        ),
    )
    parser.add_argument(
        "--azure-cli-auth",
        action="store_true",
        default=DEFAULT_USE_AZURE_CLI_AUTH,
        help="Use Azure CLI to fetch an access token for Azure SQL (pyodbc).",
    )
    parser.add_argument(
        "--table",
        default=DEFAULT_TABLE,
        help="Table name to query inside the database.",
    )
    parser.add_argument(
        "-q",
        "--question",
        required=True,
        help="Natural language question to ask about the dataset.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini"),
        help="Model name passed to ChatOpenAI.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for the language model.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    agent = SalesInsightAgent(
        db_uri=args.db_uri,
        table_name=args.table,
        model_name=args.model,
        temperature=args.temperature,
        use_azure_cli_auth=args.azure_cli_auth,
    )
    response = agent.ask(args.question)
    print(response.content)


if __name__ == "__main__":
    main()
