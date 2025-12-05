from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from agent.sales_agent import BASE_DIR as AGENT_BASE_DIR, SalesInsightAgent

DEFAULT_DB = AGENT_BASE_DIR / "data" / "retail_sales.sqlite"
DEFAULT_TABLE = "retail_sales"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="Sales Insight Agent API",
    description=(
        "Expose the LangGraph-powered SalesInsightAgent through an HTTP interface. "
        "Use the `/query` endpoint to ask dataset-aware questions and receive the agent's answer."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/artifacts", StaticFiles(directory=ARTIFACTS_DIR), name="artifacts")

class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Natural language question about the dataset.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Combined textual answer from the agent.")
    chart_path: str | None = Field(
        None, description="(Deprecated) Chart path; always null because charting is disabled."
    )


@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok", "default_db": str(DEFAULT_DB), "table": DEFAULT_TABLE}


@app.get("/", summary="Root")
def root() -> dict:
    return {
        "message": "Sales Insight Agent API",
        "default_db": str(DEFAULT_DB),
        "table": DEFAULT_TABLE,
        "docs": "/docs",
    }


@app.post("/query", response_model=QueryResponse, summary="Ask the agent a question")
def query_agent(request: QueryRequest) -> QueryResponse:
    try:
        agent = SalesInsightAgent(db_path=DEFAULT_DB, table_name=DEFAULT_TABLE)
        result = agent.ask(request.prompt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - bubble unexpected errors to clients
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(answer=result.content, chart_path=None)
