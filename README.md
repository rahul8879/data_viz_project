# Agentic Sales Insight Assistant

This repository now includes a LangGraph-powered agent that can reason over a SQLite-backed synthetic retail dataset stored in `data/retail_sales.sqlite` (table: `retail_sales`). The assistant answers natural-language questions about the dataset and automatically generates supporting charts that are saved under `artifacts/`.

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
    --db data/retail_sales.sqlite \
    --table retail_sales \
    --question "What are the total sales for the last 10 days and how do they trend daily?"
```

The agent responds with:

- A concise textual insight grounded in the numbers it computed through the `python_df` tool.
- A trend chart stored under `artifacts/` when your question includes chart-related keywords (plot/chart/visualize/graph/trend).

## 3. Serve over FastAPI

Launch the HTTP API (with Swagger UI at `/docs`) via:

```bash
uvicorn app.main:app --reload
```

Send a request using Swagger UI or cURL (only the `prompt` field is required; the app always reads `data/retail_sales.sqlite` / `retail_sales`):

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show the average sales by product"}'
```

The API response mirrors the CLI output with a concise textual answer, plus `chart_path` when a chart is generated (served from `/artifacts/...`). The agent still leverages pandas/matplotlib under the hood when needed but no longer requires charts for every question.

## 4. Web UI (React chat)

1) Start the backend: `uvicorn app.main:app --reload`
2) Install frontend deps and run the dev server:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   The Vite dev server will print a local URL (default: http://127.0.0.1:5173). The app calls the FastAPI backend at `http://127.0.0.1:8000` by default; override with `VITE_API_BASE` in a `.env` file if needed.

## 5. Bring your own SQLite table

Point the `--db` and `--table` flags to any SQLite database that includes the columns needed to answer your question (e.g., `order_date`, `revenue`, `region`). The agent automatically tries to parse date-like fields and instructs you when data is missing for a request.

## 6. Use Azure SQL with Azure CLI auth

Set the following environment variables (in `.env` or `agent/.env`) to switch the backend to Azure SQL using access tokens fetched via `az login`:

```bash
AZURE_SQL_SERVER="prod-platform-wellfit-sqlserver-reporting.database.windows.net"
AZURE_SQL_DATABASE="analytics-dev"
AZURE_SQL_TABLE="DimDate"                  # optional override; defaults to retail_sales
AZURE_SQL_DRIVER="ODBC Driver 18 for SQL Server"  # optional
AZURE_SQL_USE_CLI_AUTH=true                # uses AzureCliCredential to pass an access token to pyodbc
```

When these variables are present, the API and CLI automatically build a `mssql+pyodbc` URI and inject the Azure AD access token via `attrs_before`. Without them, the agent falls back to the bundled SQLite database.
# data_viz_project
