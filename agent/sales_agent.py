from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from operator import add
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .data_utils import (
    build_sqlite_uri,
    is_sqlite_uri,
    validate_table_name,
)

BASE_DIR = Path(__file__).resolve().parents[1]

# Accept either repo-root .env or agent/.env to match IDE setup.
for env_path in (BASE_DIR / ".env", BASE_DIR / "agent" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"Loaded environment variables from {env_path}")


class AgentState(TypedDict):
    """LangGraph state that tracks the threaded conversation."""

    messages: Annotated[List[BaseMessage], add]


@dataclass
class AgentAnswer:
    content: str


@dataclass
class SalesInsightAgent:
    """LangGraph powered agent that reasons over a SQL dataset (SQLite by default)."""

    db_path: Path
    db_uri: Optional[str] = None
    table_name: str = "retail_sales"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    # endpoint = '127.0.0.1:11434'

    def __post_init__(self) -> None:
        validate_table_name(self.table_name)

        # Default to local SQLite unless a full SQLAlchemy URI is supplied (e.g., Azure SQL).
        self.db_uri = self.db_uri or build_sqlite_uri(self.db_path)
        if is_sqlite_uri(self.db_uri) and not self.db_path.exists():
            raise FileNotFoundError(f"Could not find SQLite database at {self.db_path}")

        self.sql_database: SQLDatabase = SQLDatabase.from_uri(
            self.db_uri,
            sample_rows_in_table_info=3,
        )
        schema_snapshot = self.sql_database.get_table_info()
        self.system_message = SystemMessage(
            content=self._build_system_prompt(
                schema=schema_snapshot,
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

    def _build_system_prompt(self, schema: str) -> str:
        return (
            "You are a senior data analyst embedded inside a LangGraph agent. "
            "You have one tool: "
            "`query_sql_db` to run read-only SQL against the database (auto-detect schema, include LIMIT 50 on exploratory queries, never modify data). "
            "Follow these rules:\n"
            "- Start with `query_sql_db` to fetch the minimum data needed. Show the SQL you executed in a fenced code block before summarizing results.\n"
            "- Provide crisp, numeric answers grounded in query results. If the request is impossible with available columns, explain why and suggest the missing data.\n\n"
            "Multiple tables may exist—join them when helpful. Primary fact table: "
            f"{self.table_name}\n"
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
        description="Query a SQL table using a LangGraph-powered agent.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=BASE_DIR / "data" / "retail_sales.sqlite",
        help="Path to a SQLite database that backs the agent. Ignored if --db-uri is provided.",
    )
    parser.add_argument(
        "--db-uri",
        default=None,
        help=(
            "Full SQLAlchemy URI for non-SQLite databases (e.g., Azure SQL). "
            "Example: mssql+pyodbc://USER:PASSWORD@server.database.windows.net:1433/dbname?driver=ODBC+Driver+18+for+SQL+Server"
        ),
    )
    parser.add_argument(
        "--table",
        default="retail_sales",
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
        db_path=args.db,
        db_uri=args.db_uri,
        table_name=args.table,
        model_name=args.model,
        temperature=args.temperature,
    )
    response = agent.ask(args.question)
    print(response.content)


if __name__ == "__main__":
    main()
