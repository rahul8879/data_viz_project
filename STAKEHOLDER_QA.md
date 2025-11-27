# Stakeholder Q&A for the Sales Insight Agent

Curated talking points and answers for technology, data handling, and user experience questions.

## Product / Flow
- **What does the app do?** It lets consumers send conversational questions about sales performance and receive concise answers plus optional trend charts generated from the SQLite-backed retail dataset.
- **High-level flow?** Client (any HTTP caller) → FastAPI `/query` (or Azure Function HTTP trigger) → LangGraph agent → pandas/SQLite for computation → optional Matplotlib chart saved to `artifacts/`.
- **How do charts appear?** When the prompt includes chart keywords (plot/chart/visual/graph/trend/visualize) and relevant columns exist, the agent aggregates revenue (and profit when available), saves a timestamped PNG to `artifacts/`, and returns the path.
- **Is conversation state persisted?** Each request is independent today; messages live only per call. The agent loads fresh context from the dataset on every invocation.
- **What’s required to run?** Backend: Python 3, FastAPI (or Azure Function), LangChain/OpenAI creds, SQLite file. Azure Speech keys are optional unless a client needs microphone input.

## Data & Accuracy
- **What data source is used?** A SQLite database (`data/retail_sales.sqlite`, table `retail_sales`). If missing, the agent seeds deterministic synthetic retail data (~2,000 rows) with columns like `order_date`, `revenue`, `profit`, `discount_pct`, `region`, and `product_category`.
- **How are dates handled?** Columns with “date”/“time” in the name are auto-parsed to datetime when possible; empty or unparsable values are skipped for charting.
- **How are calculations done?** The LangGraph agent binds a `python_df` tool that runs Python against the in-memory pandas DataFrame, ensuring numeric answers come from the actual data.
- **What about data freshness?** The agent reads directly from SQLite on each request, so any updates to the table are reflected immediately without a restart.
- **Guardrails against SQL injection?** Table names are validated to alphanumeric + underscore; the agent never executes free-form SQL—aggregation is via pandas.

## Architecture & Tech Choices
- **Backend stack?** FastAPI (or Azure Function HTTP trigger) + LangGraph + LangChain `ChatOpenAI`, pandas/SQLite for data access, Matplotlib for charting. FastAPI also serves the `artifacts/` directory for generated charts.
- **Why LangGraph?** It cleanly orchestrates the LLM with tool calls (`python_df`) and defines a simple agent→tool loop, improving reliability over ad-hoc prompting.
- **Why SQLite/pandas?** Lightweight, file-based persistence that’s easy to ship; pandas provides fast aggregation/joins without needing a separate warehouse.
- **How is CORS handled?** FastAPI enables `allow_origins=["*"]` for cross-origin clients; Azure Functions can mirror this via function.json or middleware.
- **Azure Functions fit?** The `/query` logic maps directly to an HTTP-triggered function; artifacts can be written to Azure Files/Blob and served via a CDN or Static Web App.

## Deployment / Ops
- **How do I run locally?** Start backend with `uvicorn app.main:app --reload`; for Azure Functions, package the same handler as an HTTP trigger and deploy via `func azure functionapp publish`.
- **Environment variables needed?** `OPENAI_API_KEY` (LLM), optional `OPENAI_MODEL_NAME`, and for speech: `VITE_AZURE_SPEECH_KEY`, `VITE_AZURE_SPEECH_REGION`, `VITE_AZURE_SPEECH_LANG`.
- **Where are artifacts stored?** Under `artifacts/` at repo root; FastAPI serves them at `/artifacts/<filename>`.
- **How to swap datasets?** Point `--db` and `--table` (CLI) or adjust defaults in `app/main.py`; ensure the table contains necessary fields like dates and revenue/profit metrics.
- **Testing/monitoring considerations?** Add unit tests around `agent/sales_agent.py` for aggregation and chart generation; include health checks via `/health`; log chart generation paths and errors for observability.

## Risks & Limits
- **LLM dependency?** Answers rely on the configured OpenAI model; outages or quota issues will return an error message to the user.
- **Data scope?** Insights are only as good as the columns present; the agent explains when required fields are missing (e.g., no `profit` column).
- **Chart triggers?** Charts only render when prompts include chart-related keywords; otherwise users receive text-only answers.
