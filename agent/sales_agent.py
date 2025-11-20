from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from operator import add
from pathlib import Path
from typing import Annotated, List, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

BASE_DIR = Path(__file__).resolve().parents[1]

# Accept either repo-root .env or agent/.env to match IDE setup.
for env_path in (BASE_DIR / ".env", BASE_DIR / "agent" / ".env"):
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"Loaded environment variables from {env_path}")


class AgentState(TypedDict):
    """LangGraph state that tracks the threaded conversation."""

    messages: Annotated[List[BaseMessage], add]


def _summarize_dataframe(df: pd.DataFrame) -> str:
    """Return a lightweight description of the dataframe."""
    column_summaries = ", ".join(f"{col} ({dtype})" for col, dtype in df.dtypes.items())
    return (
        f"Rows: {len(df)} | Columns: {len(df.columns)}\n"
        f"Fields: {column_summaries or 'No columns discovered.'}"
    )


def _maybe_parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Convert likely date columns to datetime, ignoring failures."""
    for column in df.columns:
        if "date" in column.lower() or "time" in column.lower():
            try:
                df[column] = pd.to_datetime(df[column])
            except (ValueError, TypeError):
                continue
    return df


@dataclass
class SalesInsightAgent:
    """LangGraph powered agent that reasons over a CSV dataset."""

    csv_path: Path
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Could not find CSV at {self.csv_path}")

        self.dataframe = _maybe_parse_dates(pd.read_csv(self.csv_path))
        self.system_message = SystemMessage(
            content=self._build_system_prompt(_summarize_dataframe(self.dataframe)),
        )
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)

        python_tool = PythonAstREPLTool(
            name="python_df",
            description=(
                "Run Python code that has access to the pandas DataFrame `df` loaded "
                "from the CSV. Use this to filter, aggregate, describe, or visualize data."
            ),
            locals={
                "df": self.dataframe,
                "pd": pd,
            },
        )

        self._llm_with_tools = self.llm.bind_tools([python_tool])

        graph = StateGraph(AgentState)
        graph.add_node("agent", self._run_agent)
        graph.add_node("tools", ToolNode([python_tool]))

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        self.graph = graph.compile()

    def _build_system_prompt(self, dataset_summary: str) -> str:
        return (
            "You are a senior data analyst embedded inside a LangGraph agent. "
            "A pandas DataFrame named `df` is available through the python tool. "
            "Follow these rules:\n"
            "1. Use the python tool whenever the question requires looking at the CSV contents.\n"
            "2. Provide crisp, numeric answers grounded in the observed results. Textual guidance is fine when the user only greets you.\n"
            "3. If the requested insight is impossible with the available columns, explain why "
            "and suggest what data is needed.\n\n"
            f"Dataset snapshot:\n{dataset_summary}\n"
        )

    def _run_agent(self, state: AgentState) -> dict:
        response = self._llm_with_tools.invoke([self.system_message, *state["messages"]])
        return {"messages": [response]}

    def ask(self, question: str) -> BaseMessage:
        if not question.strip():
            raise ValueError("Question cannot be empty.")
        state = {"messages": [HumanMessage(content=question)]}
        result = self.graph.invoke(state)
        return result["messages"][-1]


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a CSV file using a LangGraph-powered agent that also builds charts.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=BASE_DIR / "data" / "sales_data.csv",
        help="Path to the CSV file that backs the agent.",
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
        csv_path=args.csv,
        model_name=args.model,
        temperature=args.temperature,
    )
    response = agent.ask(args.question)
    print(response.content)


if __name__ == "__main__":
    main()
