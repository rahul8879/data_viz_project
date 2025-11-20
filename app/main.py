from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from agent.sales_agent import BASE_DIR as AGENT_BASE_DIR, SalesInsightAgent

DEFAULT_CSV = AGENT_BASE_DIR / "data" / "sales_data.csv"

app = FastAPI(
    title="Sales Insight Agent API",
    description=(
        "Expose the LangGraph-powered SalesInsightAgent through an HTTP interface. "
        "Use the `/query` endpoint to ask dataset-aware questions and receive the agent's answer."
    ),
    version="0.1.0",
)


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Natural language question about the dataset.")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Combined textual answer from the agent.")


@app.get("/", summary="Health check")
def root() -> dict:
    return {"status": "ok", "default_csv": str(DEFAULT_CSV)}


@app.post("/query", response_model=QueryResponse, summary="Ask the agent a question")
def query_agent(request: QueryRequest) -> QueryResponse:
    try:
        agent = SalesInsightAgent(csv_path=DEFAULT_CSV)
        result = agent.ask(request.prompt)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - bubble unexpected errors to clients
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return QueryResponse(answer=result.content)
