# Architecture — AutoAnalyst MMM Copilot

## Overview

AutoAnalyst is a two-LLM-call, three-stage pipeline wrapped in a dual interface (Gradio UI + FastAPI). Each question triggers:

1. **Planner** — LLM decides which tools to call and in what order
2. **Executor** — Pure Python runs those tools (no LLM)
3. **Synthesizer** — LLM converts tool results into a structured analyst-style answer

This pattern keeps reasoning separated from execution. The executor has no LLM dependency, making it fast, deterministic, and easy to test.

---

## Component Map

```
┌──────────────────────────────────────────────────────────┐
│                        Interfaces                        │
│                                                          │
│   app.py (Gradio)          main.py (FastAPI)            │
│   • /question textbox      • POST /analyze               │
│   • streaming status       • POST /query                 │
│   • markdown output        • GET  /health                │
│   • image chart            • Pydantic request/response   │
└────────────────┬────────────────────┬────────────────────┘
                 │                    │
                 └────────┬───────────┘
                          │  pipeline.run(question, collection)
                          ▼
┌──────────────────────────────────────────────────────────┐
│                     pipeline.py                          │
│                                                          │
│  run_planner(question) → TaskPlan                        │
│    └─ LiteLLM call → JSON → Pydantic TaskPlan            │
│                                                          │
│  run_executor(plan, tool_registry) → execution_log       │
│    └─ Pure Python loop, no LLM                           │
│    └─ Calls each tool function in order                  │
│                                                          │
│  run_synthesizer(question, execution_log) → dict         │
│    └─ LiteLLM call → JSON → CopilotAnswer                │
└──────────────────────────────────────────────────────────┘
          │
          ▼ tool_registry
┌──────────────────────────────────────────────────────────┐
│                      src/ modules                        │
│                                                          │
│  db.py              ChromaDB init, embedding function    │
│  query_parser.py    mmm_retriever (regex + LLM fallback) │
│  benchmark_fetcher  benchmark_fetcher, benchmark_all     │
│  scenario_simulator scenario_simulator, simulate_portfolio│
│  chart_generator    6 chart functions → base64 PNG       │
└──────────────────────────────────────────────────────────┘
          │
          ▼
┌──────────────────────────────────────────────────────────┐
│               Data & Infrastructure                      │
│                                                          │
│  mmm_vectorstore/   ChromaDB persistent files            │
│  mmm_dummy_data/    50 synthetic MMM JSON model outputs  │
│  .env               LLM API keys (never committed)       │
└──────────────────────────────────────────────────────────┘
```

---

## Pipeline Detail

### Stage 1 — Planner

**Input**: user question (string)  
**Output**: `TaskPlan` (Pydantic model)

```python
class SubTask(BaseModel):
    task:      str       # plain English description
    tool_name: str       # name matching tool_registry key
    tool_args: dict      # args passed verbatim to the tool

class TaskPlan(BaseModel):
    objective: str
    subtasks:  list[SubTask]
    reasoning: str
```

The planner system prompt lists all 9 available tools with their argument schemas. Key rules enforced in the prompt:
- Always start with `mmm_retriever`
- Include `scenario_simulator` only if the question involves budget changes
- Include `benchmark_fetcher` only if the question asks about industry comparison
- End with exactly one chart tool

**Fallback**: if LLM output is not valid JSON, falls back to a single `mmm_retriever` call.

### Stage 2 — Executor

**Input**: `TaskPlan`, `tool_registry` dict  
**Output**: `execution_log` (list of dicts: task, tool_name, result)

Pure Python loop — no LLM calls. Each subtask:
1. Looks up the tool function by `tool_name`
2. Calls `tool_fn(subtask.tool_args)`
3. Appends result to the log

Chart tools return base64-encoded PNG strings. All other tools return dicts or strings.

Errors are caught per-subtask and logged as `ERROR: ...` strings so the synthesizer can acknowledge missing data gracefully.

### Stage 3 — Synthesizer

**Input**: original question, `execution_log`  
**Output**: `CopilotAnswer` dict + `chart_b64`

Builds a context string from all tool results (truncated at 3000 chars each to stay within context limits). Chart base64 blobs are extracted separately and passed through outside the LLM context.

Returns structured JSON matching `CopilotAnswer`:
```python
class CopilotAnswer(BaseModel):
    summary:   str                  # 2-3 sentence executive summary
    insights:  list[ChannelInsight] # per-channel ROI comparison table
    narrative: str                  # 3-5 paragraph detailed analysis
    chart_type: Optional[str]       # which chart was generated
    sources:   list[str]            # brands referenced
```

---

## Query Parser

Two-stage query filter extraction (`src/query_parser.py`):

```
Question → regex_parser() → filters dict
                │
                ├── found filters? → use them → build_where_clause() → ChromaDB where
                │
                └── empty? → llm_parser() → same filters dict → build_where_clause()
```

Filters extracted: `brands`, `year`, `channel`, `category` (hcp/consumer), `sub_vertical` (oncology/vaccines/pharma).

The `build_where_clause` function converts these into ChromaDB `$eq`/`$in`/`$and` filter syntax.

---

## Vector Store

**Engine**: ChromaDB (persistent, local)  
**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)  
**Collection**: `mmm_outputs`  
**Document count**: ~50 MMM model JSON outputs, each split into one document per channel

Each document metadata includes: `brand`, `sub_vertical`, `year`, `channel`, `category`, `type`.

The collection is loaded once at startup and passed through the call chain to avoid re-initialization per request.

---

## Benchmark & Scenario Data

### Benchmarks (`src/benchmark_fetcher.py`)

Static lookup table: channel → sub_vertical → `{roi_low, roi_mid, roi_high, percentile_75}`.

Covers 14 channels across oncology/vaccines/pharma. Source: aggregated from 80+ pharma MMM studies (2019–2024).

The `benchmark_fetcher()` function returns a performance label (outperformer / average / underperformer), gap to median, gap to 75th percentile, and a plain-English insight string for the synthesizer.

### Scenario Simulator (`src/scenario_simulator.py`)

Uses a power curve model: `new_revenue = current_revenue × (new_spend / current_spend)^exponent`

Power curve exponents per channel (lower = stronger diminishing returns):

| Channel | Exponent | Interpretation |
|---------|----------|---------------|
| `paid_search` | 0.8 | Near-linear; scales efficiently |
| `social`, `display`, `pulsepoint` | 0.6 | Moderate returns |
| `doximity`, `medscape`, `sfmc`, `nexgen` | 0.5 | Standard diminishing returns |
| `tv`, `streaming_tv` | 0.4 | Heavy saturation |
| `salesforce_calls` | 0.3 | Strongest diminishing returns |

Both single-channel and portfolio-level (multi-channel) simulation are supported.

---

## Chart Types

All charts render via matplotlib with a dark theme (`#0F172A` background) and return base64-encoded PNG.

| Function | Chart Type | Use Case |
|----------|-----------|---------|
| `roi_bar_chart` | Horizontal bar | ROI ranking across channels |
| `spend_vs_revenue_chart` | Scatter (bubble = ROI) | Efficiency vs volume view |
| `revenue_waterfall_chart` | Stacked waterfall | Revenue decomposition (base + incremental) |
| `scenario_bar_chart` | Grouped bar | Before/after budget scenario |
| `benchmark_comparison_chart` | Grouped bar (3 series) | Your ROI vs median vs 75th percentile |
| `power_curve_chart` | Line + scatter | Diminishing returns curve with current position |

---

## Configuration

All tuneable constants live in `src/config.py`:

| Constant | Default | Override via |
|---------|---------|-------------|
| `MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Edit config |
| `COLLECTION_NAME` | `mmm_outputs` | Edit config |
| `VECTORSTORE_PATH` | `./mmm_vectorstore` | Edit config |
| `DATA_FOLDER` | `mmm_dummy_data` | Edit config |
| `LLM_MODEL` | `gemini/gemini-3.1-flash-lite` | Edit config or `LLM_MODEL` env var |

LiteLLM is used as the LLM abstraction layer, so `LLM_MODEL` can be set to any supported provider string (e.g., `openai/gpt-4o`, `anthropic/claude-3-5-sonnet-20241022`).

---

## Known Limitations

See `docs/faang_audit.md` for a full production-readiness audit. Key items:

- LLM calls in FastAPI are synchronous (blocks event loop under concurrent load)
- No authentication on API endpoints
- No rate limiting
- No test coverage
- `LLM_MODEL` default references a non-existent model version — update to a valid model ID
- Power curve exponents and benchmark data are hardcoded — no runtime config reload
