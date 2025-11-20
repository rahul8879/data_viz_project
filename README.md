# Agentic Sales Insight Assistant

This repository now includes a LangGraph-powered agent that can reason over the CSV data stored in `data/sales_data.csv`. The assistant answers natural-language questions about the dataset and automatically generates supporting charts that are saved under `artifacts/`.

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Export an OpenAI API key so the LangChain `ChatOpenAI` client can authenticate:

```bash
export OPENAI_API_KEY="sk-..."
```

## 2. Ask a question

```bash
python -m agent.sales_agent \
    --csv data/sales_data.csv \
    --question "What are the total sales for the last 10 days and how do they trend daily?"
```

The agent responds with:

- A concise textual insight grounded in the numbers it computed through the `python_df` tool.
- The path to a chart stored under `artifacts/`. Each invocation wipes nothing, so you can inspect or embed the produced visual later.

## 3. Serve over FastAPI

Launch the HTTP API (with Swagger UI at `/docs`) via:

```bash
uvicorn app.main:app --reload
```

Send a request using Swagger UI or cURL (only the `prompt` field is required; the app always reads `data/sales_data.csv`):

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show the average sales by product"}'
```

The API response mirrors the CLI output with a concise textual answer. The agent still leverages pandas/matplotlib under the hood when needed but no longer requires charts for every question.

## 3. Bring your own CSV

Point the `--csv` flag to any other dataset that includes the columns needed to answer your question (e.g., `date`, `sales_amount`, `region`). The agent automatically tries to parse date-like fields and instructs you when data is missing for a request.
# data_viz_project
