# AutoAnalyst — Pharma MMM Copilot

An AI-powered Marketing Mix Modeling (MMM) analysis platform for pharmaceutical brands. Ask natural language questions about channel ROI, budget scenarios, industry benchmarks, and spend efficiency — get structured answers with auto-generated charts.

## What It Does

- **Answers MMM questions in plain English** — e.g. "Which HCP channels are underperforming for Keytruda?"
- **Runs budget scenarios** — simulates revenue impact of spend shifts across channels using calibrated power curves
- **Benchmarks against industry data** — compares your ROI to aggregated pharma MMM benchmarks (80+ brands, 2019–2024)
- **Generates charts automatically** — ROI bars, waterfall decompositions, scenario comparisons, diminishing return curves
- **Supports oncology, vaccines, and pharma** sub-verticals with channel-specific benchmark data

## Architecture

```
User question
     │
     ▼
┌─────────────┐     LLM call #1     ┌─────────────────────┐
│   Planner   │ ──────────────────► │  TaskPlan (JSON)    │
│  (pipeline) │                     │  objective          │
└─────────────┘                     │  subtasks[]         │
                                    │  reasoning          │
                                    └─────────────────────┘
                                             │
                                             ▼
┌─────────────┐   pure Python loop  ┌─────────────────────┐
│  Executor   │ ──────────────────► │  Tool Results       │
│  (pipeline) │                     │  • mmm_retriever    │
└─────────────┘                     │  • benchmark_fetch  │
                                    │  • scenario_sim     │
                                    │  • chart generators │
                                    └─────────────────────┘
                                             │
                                             ▼
┌─────────────┐     LLM call #2     ┌─────────────────────┐
│ Synthesizer │ ◄───────────────── │  context bundle     │
│  (pipeline) │                     └─────────────────────┘
└─────────────┘
     │
     ▼
CopilotAnswer { summary, insights[], narrative, chart_b64, sources }
     │
     ├── Gradio UI  (app.py)
     └── FastAPI    (main.py)
```

**Vector store**: ChromaDB with `all-MiniLM-L6-v2` sentence-transformer embeddings over 50 pharma MMM model outputs.

**Query parsing**: Regex parser first; falls back to LLM-based extraction if no structured filters are found.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (package manager)
- API key for your LLM provider (default: Gemini via LiteLLM)

### Install

```bash
git clone <repo>
cd auto-analyst
uv sync
```

### Environment Variables

Create a `.env` file (never commit it):

```env
# LLM provider key — LiteLLM supports OpenAI, Gemini, Anthropic, etc.
GEMINI_API_KEY=your_key_here

# Optional overrides (defaults shown)
# LLM_MODEL=gemini/gemini-2.0-flash-lite
# VECTORSTORE_PATH=./mmm_vectorstore
# DATA_FOLDER=mmm_dummy_data
```

### Run the Gradio UI

```bash
uv run python app.py
# Opens at http://localhost:7860
```

### Run the FastAPI Server

```bash
uv run uvicorn main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### Run as CLI

```bash
uv run python pipeline.py
# Runs 3 sample questions and prints results to stdout
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check — returns `{"status": "healthy"}` |
| `GET`  | `/` | Root — returns API name |
| `POST` | `/query` | Raw semantic retrieval from vector store |
| `POST` | `/analyze` | Full pipeline: plan → execute tools → synthesize answer |

### `POST /analyze`

```json
// Request
{ "question": "What is the ROI of HCP channels for oncology brands?" }

// Response
{
  "summary": "...",
  "insights": [
    {
      "channel": "doximity",
      "current_roi": 2.9,
      "benchmark_roi": 3.0,
      "recommendation": "...",
      "confidence": "high"
    }
  ],
  "narrative": "...",
  "chart_type": "roi_bar",
  "sources": ["keytruda", "lenvima"],
  "chart_b64": "<base64 PNG>"
}
```

### `POST /query`

```json
// Request
{ "question": "Keytruda paid search performance 2024", "n_results": 10 }

// Response
{ "question": "...", "result": "--- Result 1 ---\nBrand: keytruda | ..." }
```

## Pipeline Tools

The planner selects from these tools to answer each question:

| Tool | Purpose | Key Args |
|------|---------|---------|
| `mmm_retriever` | Semantic search over vector store | `question`, `n_results` |
| `benchmark_fetcher` | Compare channel ROI vs industry benchmarks | `channel`, `sub_vertical`, `current_roi` |
| `scenario_simulator` | Simulate budget change revenue impact | `channel`, `current_spend`, `current_revenue_contribution`, `budget_change_pct` |
| `roi_bar_chart` | Horizontal bar chart of channel ROIs | `channels[]`, `brand`, `sub_vertical` |
| `spend_vs_revenue_chart` | Scatter: spend vs revenue (bubble = ROI) | `channels[]`, `brand` |
| `revenue_waterfall_chart` | Base + incremental revenue decomposition | `model_summary`, `channels[]` |
| `scenario_bar_chart` | Before/after revenue grouped bar chart | `scenario_results[]` |
| `benchmark_comparison_chart` | Your ROI vs median vs 75th percentile | `benchmark_results[]`, `sub_vertical` |
| `power_curve_chart` | Diminishing returns curve with current position | `channel`, `current_spend`, `current_revenue` |

## Example Questions

```
What is the ROI of HCP channels for oncology brands and how do they compare to industry benchmarks?

If I cut TV budget by 20% for vaccines and reinvest it in paid search, what happens to revenue?

Which channels are underperforming for Keytruda and what should we do about them?

Show me the spend vs revenue breakdown for oncology brands in 2024.

What are the diminishing returns on Salesforce calls for oncology?
```

## Data

### Vector Store

50 synthetic MMM model JSON files in `mmm_dummy_data/`, covering:

- **Sub-verticals**: oncology, vaccines, pharma
- **Brands**: Keytruda, Lenvima, Welireg, Qliftara, Januvia, Belsomra, Bridion, Verquovo, Dificid, Zerbaxia, Del-PIF, RotaTeq, Vaxneuvance, Vaqta, Vaxelis, ProQuad, Capvaxive, Gardasil
- **Channels**: salesforce_calls, doximity, medscape, pulsepoint, deepintent, sfmc, nexgen, tv, streaming_tv, online_video, paid_search, social, display, audio

Each JSON file contains spend, ROI, revenue contribution, adstock parameters, and model fit statistics per channel.

### Benchmark Data

Hardcoded benchmarks in `src/benchmark_fetcher.py` — aggregated from 80+ pharma MMM studies (2019–2024). Covers ROI low/mid/high/p75 per channel per sub-vertical, plus engagement metrics (open rates, CTR) for email and search channels.

### Scenario Simulator

Power curve exponents in `src/scenario_simulator.py` calibrated per channel:
- `paid_search`: 0.8 (near-linear, scales efficiently)
- `salesforce_calls`: 0.3 (strong diminishing returns)
- Most digital channels: 0.5–0.6

## Docker

```bash
# Build
docker build -t auto-analyst .

# Run FastAPI server
docker run -p 8000:8000 --env-file .env auto-analyst

# Run Gradio UI instead
docker run -p 7860:7860 --env-file .env auto-analyst \
  .venv/bin/python app.py
```

## Project Structure

```
auto-analyst/
├── app.py                   # Gradio UI
├── main.py                  # FastAPI server
├── pipeline.py              # Core pipeline (Planner → Executor → Synthesizer)
├── src/
│   ├── config.py            # Constants: model names, paths, year vars
│   ├── db.py                # ChromaDB vector store init
│   ├── query_parser.py      # Regex + LLM query filter extraction
│   ├── benchmark_fetcher.py # Industry benchmark data and comparisons
│   ├── scenario_simulator.py# Budget scenario simulation with power curves
│   ├── chart_generator.py   # 6 matplotlib chart types → base64 PNG
│   ├── embedder.py          # Data ingestion: JSON → ChromaDB
│   └── generate_mmm_output.py # Synthetic MMM data generator
├── mmm_dummy_data/          # 50 synthetic MMM model JSON files
├── mmm_vectorstore/         # ChromaDB persistent store (auto-created)
├── docs/                    # Architecture and audit documentation
├── dockerfile
└── pyproject.toml
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `gradio` | Web UI |
| `fastapi` + `uvicorn` | REST API |
| `chromadb` | Vector store |
| `sentence-transformers` | Text embeddings |
| `litellm` | LLM provider abstraction |
| `pydantic` | Structured output validation |
| `matplotlib` + `numpy` | Chart generation |
| `python-dotenv` | Environment variable loading |
