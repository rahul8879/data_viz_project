from __future__ import annotations

import argparse
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from operator import add
from pathlib import Path
from typing import Annotated, List, Optional, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.utilities import SQLDatabase
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


@dataclass
class AgentAnswer:
    content: str
    chart_path: Optional[Path]


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
    """LangGraph powered agent that reasons over a SQLite-backed retail dataset."""

    db_path: Path
    table_name: str = "retail_sales"
    artifacts_dir: Path = BASE_DIR / "artifacts"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    # endpoint = '127.0.0.1:11434'

    def __post_init__(self) -> None:
        _validate_table_name(self.table_name)
        ensure_sqlite_dataset(self.db_path, self.table_name)
        self.sql_database: SQLDatabase = SQLDatabase.from_uri(
            f"sqlite:///{self.db_path}",
            sample_rows_in_table_info=3,
        )
        self.dataframe = _maybe_parse_dates(self._load_dataframe())
        schema_snapshot = self.sql_database.get_table_info()
        self.system_message = SystemMessage(
            content=self._build_system_prompt(
                dataset_summary=_summarize_dataframe(self.dataframe),
                schema=schema_snapshot,
            ),
        )
        self.llm = ChatOpenAI(model=self.model_name, temperature=self.temperature)
        # self.llm = ChatOpenAI(model="Gemma3-1B", temperature=self.temperature,endpoints= endpoint)
   
        python_tool = PythonAstREPLTool(
            name="python_df",
            description=(
                "Run Python code that has access to the pandas DataFrame `df` loaded "
                "from the SQLite table. Use this to filter, aggregate, describe, or visualize data."
            ),
            locals={
                "df": self.dataframe,
                "pd": pd,
            },
        )

        sql_tool = QuerySQLDataBaseTool(db=self.sql_database)

        self._llm_with_tools = self.llm.bind_tools([python_tool, sql_tool])

        graph = StateGraph(AgentState)
        graph.add_node("agent", self._run_agent)
        graph.add_node("tools", ToolNode([python_tool, sql_tool]))

        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        self.graph = graph.compile()

    def _build_system_prompt(self, dataset_summary: str, schema: str) -> str:
        return (
            "You are a senior data analyst embedded inside a LangGraph agent. "
            "You have two tools: "
            "1) `query_sql_db` to run read-only SQL against the SQLite database (auto-detect schema, include LIMIT 50 on exploratory queries, never modify data); "
            "2) `python_df` to operate on the pandas DataFrame `df` already loaded from the primary fact table for extra calculations or chart prep. "
            "Follow these rules:\n"
            "- Start with `query_sql_db` to fetch the minimum data needed. Show the SQL you executed in a fenced code block before summarizing results.\n"
            "- Use `python_df` only when SQL cannot easily answer the request (e.g., complex visuals or custom calculations).\n"
            "- Provide crisp, numeric answers grounded in query results. If the request is impossible with available columns, explain why and suggest the missing data.\n\n"
            "Multiple tables may exist—join them when helpful. Primary fact table: "
            f"{self.table_name}\n"
            f"SQL schema snapshot:\n{schema}\n\n"
            f"Dataset snapshot:\n{dataset_summary}\n"
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
        chart_path = self._maybe_generate_chart(question)
        return AgentAnswer(content=final_message.content, chart_path=chart_path)

    def _load_dataframe(self) -> pd.DataFrame:
        if not self.db_path.exists():
            raise FileNotFoundError(f"Could not find SQLite database at {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            _validate_table_name(self.table_name)
            return pd.read_sql(f"SELECT * FROM {self.table_name}", conn)

    def _maybe_generate_chart(self, question: str) -> Optional[Path]:
        keywords = ("plot", "chart", "visual", "graph", "trend", "visualize")
        if not any(k in question.lower() for k in keywords):
            return None
        if not {"order_date", "revenue"}.issubset(self.dataframe.columns):
            return None

        df = self.dataframe.copy()
        df = _maybe_parse_dates(df)
        df = df.dropna(subset=["order_date", "revenue"])
        if df.empty:
            return None

        aggregations = {"revenue": ("revenue", "sum")}
        if "profit" in df.columns:
            aggregations["profit"] = ("profit", "sum")
        daily = (
            df.groupby(pd.to_datetime(df["order_date"]).dt.date)
            .agg(**aggregations)
            .reset_index()
            .sort_values("order_date")
        )
        if daily.empty:
            return None

        _make_artifacts_dir(self.artifacts_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chart_path = self.artifacts_dir / f"trend_{timestamp}.png"

        plt.figure(figsize=(8, 4))
        plt.plot(daily["order_date"], daily["revenue"], label="Revenue", color="#2f80ed")
        if "profit" in daily.columns:
            plt.plot(daily["order_date"], daily["profit"], label="Profit", color="#52c41a", alpha=0.85)
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.ylabel("Amount")
        plt.title("Revenue & Profit Trend")
        plt.tight_layout()
        plt.legend()
        plt.grid(alpha=0.2)
        plt.savefig(chart_path, dpi=150)
        plt.close()
        return chart_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Query a SQLite table using a LangGraph-powered agent that also builds charts.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=BASE_DIR / "data" / "retail_sales.sqlite",
        help="Path to the SQLite database that backs the agent. The table is auto-created with synthetic data if missing.",
    )
    parser.add_argument(
        "--table",
        default="retail_sales",
        help="Table name inside the SQLite database.",
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
        table_name=args.table,
        model_name=args.model,
        temperature=args.temperature,
    )
    response = agent.ask(args.question)
    print(response.content)
    if response.chart_path:
        print(f"Chart saved to: {response.chart_path}")


if __name__ == "__main__":
    main()


def ensure_sqlite_dataset(db_path: Path, table_name: str = "retail_sales", rows: int = 2000) -> None:
    """Create a synthetic retail sales dataset in SQLite if it does not already exist."""
    _validate_table_name(table_name)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        # Create the table only when it is absent to avoid clobbering user-provided data.
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
        existing_tables = set(tables["name"].values)
        source_df: pd.DataFrame | None = None

        if table_name not in existing_tables:
            source_df = _generate_retail_sales_data(rows)
            source_df.to_sql(table_name, conn, index=False, if_exists="replace")
            existing_tables.add(table_name)
        else:
            # Pull a lightweight sample to seed dimension tables if needed.
            try:
                source_df = pd.read_sql(f"SELECT * FROM {table_name} LIMIT 500", conn)
            except Exception:
                source_df = None

        _ensure_dimension_tables(conn, table_name, seed_df=source_df)


def _validate_table_name(table_name: str) -> None:
    """Guard against SQL injection inside table names."""
    if not table_name.replace("_", "").isalnum():
        raise ValueError("Table names may only contain letters, numbers, and underscores.")


def _generate_retail_sales_data(rows: int) -> pd.DataFrame:
    """Generate deterministic synthetic retail sales data."""
    rng = np.random.default_rng(42)

    start_date = datetime.now().date() - timedelta(days=365)
    product_catalog = {
        "Electronics": ["Smartphone", "Laptop", "Headphones", "Tablet", "Smartwatch"],
        "Home": ["Vacuum Cleaner", "Coffee Maker", "Air Purifier", "Blender", "Air Fryer"],
        "Grocery": ["Organic Apples", "Almond Milk", "Pasta", "Olive Oil", "Granola Bars"],
        "Clothing": ["Denim Jeans", "Hoodie", "Sneakers", "Jacket", "T-Shirt"],
        "Beauty": ["Moisturizer", "Shampoo", "Perfume", "Lipstick", "Serum"],
    }
    regions = ["North", "South", "East", "West", "Central"]
    channels = ["In-Store", "Online", "Mobile App", "Wholesale"]
    segments = ["Loyalty", "New", "Returning", "Corporate"]

    categories = rng.choice(list(product_catalog.keys()), size=rows)
    products = [rng.choice(product_catalog[cat]) for cat in categories]
    store_ids = rng.integers(100, 120, size=rows)
    order_dates = [start_date + timedelta(days=int(day)) for day in rng.integers(0, 365, size=rows)]
    units = rng.integers(1, 12, size=rows)
    base_prices = {
        "Electronics": 350,
        "Home": 120,
        "Grocery": 8,
        "Clothing": 60,
        "Beauty": 40,
    }
    unit_prices = np.array([base_prices[cat] for cat in categories]) * rng.uniform(0.85, 1.25, size=rows)
    discount_pct = rng.choice([0, 0.05, 0.1, 0.15, 0.2], p=[0.35, 0.25, 0.2, 0.15, 0.05], size=rows)
    unit_costs = unit_prices * rng.uniform(0.45, 0.7, size=rows)

    revenue = units * unit_prices * (1 - discount_pct)
    cost = units * unit_costs
    profit = revenue - cost

    df = pd.DataFrame(
        {
            "order_id": np.arange(1, rows + 1),
            "order_date": order_dates,
            "store_id": store_ids,
            "region": rng.choice(regions, size=rows),
            "channel": rng.choice(channels, size=rows),
            "customer_segment": rng.choice(segments, size=rows),
            "product_category": categories,
            "product_name": products,
            "units_sold": units,
            "unit_price": unit_prices.round(2),
            "discount_pct": discount_pct.round(2),
            "unit_cost": unit_costs.round(2),
            "revenue": revenue.round(2),
            "cost": cost.round(2),
            "profit": profit.round(2),
        }
    )
    return df


def _ensure_dimension_tables(conn: sqlite3.Connection, fact_table: str, seed_df: pd.DataFrame | None) -> None:
    """Create small dimension tables to exercise multi-table SQL if missing."""
    tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table'", conn)
    existing = set(tables["name"].values)
    rng = np.random.default_rng(7)

    if "store_info" not in existing:
        if seed_df is None or seed_df.empty:
            store_ids = np.arange(100, 110)
            regions = rng.choice(["North", "South", "East", "West", "Central"], size=len(store_ids))
            channels = rng.choice(["In-Store", "Online", "Mobile App", "Wholesale"], size=len(store_ids))
        else:
            dedup = seed_df[["store_id", "region", "channel"]].drop_duplicates()
            store_ids = dedup["store_id"].values
            regions = dedup["region"].values
            channels = dedup["channel"].values

        managers = [f"Manager {i+1}" for i in range(len(store_ids))]
        opened = [
            (datetime.now().date() - timedelta(days=int(offset))).isoformat()
            for offset in rng.integers(180, 900, size=len(store_ids))
        ]
        store_info = pd.DataFrame(
            {
                "store_id": store_ids,
                "region": regions,
                "channel": channels,
                "manager": managers,
                "opened_date": opened,
            }
        )
        store_info.to_sql("store_info", conn, index=False, if_exists="replace")

    if "product_catalog" not in existing:
        if seed_df is None or seed_df.empty or not {"product_name", "product_category", "unit_price"}.issubset(
            seed_df.columns
        ):
            product_catalog = pd.DataFrame(
                {
                    "product_name": ["Starter Widget", "Pro Widget", "Elite Widget", "Accessory Pack"],
                    "product_category": ["Electronics", "Electronics", "Electronics", "Accessories"],
                    "list_price": [129.0, 249.0, 399.0, 59.0],
                }
            )
        else:
            product_catalog = (
                seed_df.groupby(["product_name", "product_category"], as_index=False)["unit_price"]
                .mean()
                .rename(columns={"unit_price": "list_price"})
                .sort_values("product_name")
            )
        product_catalog.to_sql("product_catalog", conn, index=False, if_exists="replace")


def _make_artifacts_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
