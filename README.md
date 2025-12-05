# Agentic Sales Insight Assistant (Azure SQL)

Conversational agent that reads directly from Azure SQL (no SQLite fallback). It authenticates via Azure CLI and uses LangGraph + LangChain to answer questions over your tables.

## 1) Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # required for the LLM
```

## 2) Configure Azure SQL (required)
Set these in `.env` or `agent/.env`:
```bash
AZURE_SQL_SERVER="prod-platform-wellfit-sqlserver-reporting.database.windows.net"
AZURE_SQL_DATABASE="analytics-dev"
AZURE_SQL_TABLE="DimDate"                  # optional override
AZURE_SQL_DRIVER="ODBC Driver 17 for SQL Server"  # optional
AZURE_SQL_USERNAME="your_user"
AZURE_SQL_PASSWORD="your_password"
# Optional: set AZURE_SQL_USE_CLI_AUTH=true to use AzureCliCredential instead of username/password
# If you enable CLI auth, run: az login
```

## 3) Ask a question (CLI)
```bash
python -m agent.sales_agent --question "Show the last 10 dates and their flags"
```
By default it builds a `mssql+pyodbc` URI from the env vars and injects the Azure AD token via `attrs_before`. Override with `--db-uri` if you want to supply your own pyodbc connection string.

## 4) Serve over FastAPI
```bash
uvicorn app.main:app --reload
```
Call the API:
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Show the top 5 dates"}'
```
Health check: `http://127.0.0.1:8000/health`

## 5) React chat UI
```bash
cd frontend
npm install
npm run dev
```
The frontend defaults to `http://127.0.0.1:8000`; override with `VITE_API_BASE` in `frontend/.env` if needed.
