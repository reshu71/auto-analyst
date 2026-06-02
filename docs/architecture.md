# MMM Copilot — System Architecture

## Overview

A pharma Marketing Mix Modeling (MMM) copilot that answers commercial analytics questions using a three-stage AI pipeline: plan → execute → synthesize. The system combines vector-search over MMM model outputs with rule-based simulators and benchmark databases, all instrumented with Langfuse v4 distributed tracing.

---

## High-Level Flow

```
User Question
     │
     ▼
┌─────────────┐      LLM (litellm)
│   Planner   │ ──────────────────► TaskPlan (ordered SubTask list)
└─────────────┘
     │
     ▼
┌─────────────┐
│  Executor   │ ── loops over SubTasks ──► Tool Registry
└─────────────┘                                │
     │                        ┌────────────────┴──────────────────┐
     │                        │                                   │
     │                  mmm_retriever                   benchmark_fetcher
     │                  (agentic RAG)                  scenario_simulator
     │                        │                           chart_*
     ▼
┌─────────────┐      LLM (litellm)
│ Synthesizer │ ──────────────────► CopilotAnswer (JSON)
└─────────────┘
     │
     ▼
  Answer + Chart (base64 PNG)
```

---

## Components

### `pipeline.py` — Core Pipeline

The main entry point. Exposes `run(question, collection)` which:

1. Builds the tool registry bound to the current ChromaDB collection
2. Calls `run_planner()` → `TaskPlan`
3. Calls `run_executor()` → `list[dict]` execution log
4. Calls `run_synthesizer()` → `dict` answer

Every function is wrapped with `@observe` (Langfuse v4) — planner and synthesizer are `as_type="generation"` (token tracking); executor and tools are spans.

**Pydantic Models:**
- `SubTask` — `{ task, tool_name, tool_args }`
- `TaskPlan` — `{ objective, subtasks, reasoning }`
- `ChannelInsight` — per-channel ROI + benchmark + recommendation
- `CopilotAnswer` — `{ summary, insights, narrative, chart_type, sources }`

---

### `src/agentic_retriver.py` — Iterative RAG Retriever

Replaces single-shot embedding lookup with a 3-step loop (up to `max_retries`):

1. **Query rewriter** — LLM rewrites the question as a dense keyword string optimised for semantic search; on retry, targets the identified gap
2. **Keyword check** — fast string matching to confirm retrieved chunks mention expected brands/channels/year
3. **LLM-as-judge** — deep sufficiency check; if insufficient, updates gap and retries

Falls back gracefully: returns best results seen, or a `did_you_mean` suggestion on zero results.

**Key function:** `agentic_retriever(question, collection, n_results=10, max_retries=3)`

---

### `src/query_parser.py` — Filter Extraction

Extracts structured ChromaDB `where` filters from free-text questions.

- `regex_parser` — fast brand/channel/year/vertical detection via regex
- `llm_parser` — fallback LLM call when regex finds nothing
- `build_where_clause(filters, question)` — converts filters to ChromaDB `$and`/`$eq`/`$in` syntax; appends `{"type": "summary"}` for model-level questions

---

### `src/benchmark_fetcher.py` — Industry Benchmarks

Static lookup table of ROI benchmark ranges (low / mid / 75th pct / high) for 14 channels × 3 sub-verticals (oncology / vaccines / pharma), sourced from 100+ pharma MMM studies.

- `benchmark_fetcher(channel, sub_vertical, current_roi=None)` — single channel
- `benchmark_all_channels(channels, sub_vertical)` — batch

When `current_roi` is omitted the function returns ranges only; the Synthesizer performs the comparison against retrieved brand ROI.

---

### `src/scenario_simulator.py` — Budget Simulation

Power-curve model: `new_revenue = current_revenue × (new_spend / current_spend) ^ exponent`

Exponents are calibrated per channel (0.3 for Salesforce calls, 0.8 for paid search). Provides before/after spend, revenue, ROI, and a plain-English efficiency note.

- `scenario_simulator(input_json)` — single channel
- `simulate_portfolio(channels, budget_changes)` — multi-channel with blended ROI

---

### `src/chart_generator.py` — Visualisation

Six chart types, all rendered server-side (Matplotlib Agg) and returned as base64 PNG strings:

| Function | Chart type |
|---|---|
| `roi_bar_chart` | Horizontal bar — ROI per channel |
| `spend_vs_revenue_chart` | Scatter — spend vs revenue contribution |
| `revenue_waterfall_chart` | Waterfall — revenue decomposition |
| `scenario_bar_chart` | Grouped bar — before/after scenario |
| `benchmark_comparison_chart` | Grouped bar — brand ROI vs industry benchmarks |
| `power_curve_chart` | Line — diminishing returns curve |

HCP channels render in blue (`#2563EB`), consumer channels in teal (`#0D9488`), against a dark background (`#0F172A`).

---

### `src/db.py` — Vector Store

Thin wrapper around ChromaDB's `PersistentClient`. Creates or connects to a `sentence-transformers/all-MiniLM-L6-v2` embedded collection at `./mmm_vectorstore`.

---

### `src/embedder.py` — Data Ingestion

Chunks MMM JSON output files from `mmm_dummy_data/` into two chunk types per brand-year:
- **summary** — model-level metrics (R², MAPE, base/incremental split)
- **channel** — per-channel metrics (spend, ROI, reach, tactics)

Skips already-ingested IDs; idempotent.

---

### `src/graph/` — LangGraph Pipeline

An alternative execution layer that wraps the core pipeline in a LangGraph `StateGraph` for human-in-the-loop approval and checkpointing.

```
planner ──► human_approval ──(approve)──► executor ──► synthesizer ──► END
                            ──(reject) ──► rejected                 ──► END
```

**Files:**
- `state.py` — `MMMGraphState` (TypedDict with reducers) + `GraphConfig`
- `nodes.py` — one function per node; calls pipeline functions internally
- `routing.py` — pure predicate functions for conditional edges
- `graph.py` — `build_graph_with_approval()` and `build_graph_silent()` factories
- `runner.py` — `run_with_approval()` and `run_silent()` entry points (SQLite checkpointer)

---

### `src/config.py` — Global Constants

```python
MODEL_NAME       = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME  = "mmm_outputs"
VECTORSTORE_PATH = "./mmm_vectorstore"
LLM_MODEL        = "gemini/gemini-3.1-flash-lite"
```

---

## Observability — Langfuse v4

Every pipeline run produces a nested trace:

```
mmm_pipeline  [trace]
  ├── planner       [generation]  model + token + cost auto-tracked
  ├── executor      [span]
  │     ├── mmm_retriever          [span]
  │     ├── benchmark_fetcher      [span]
  │     └── <chart tool>           [span]
  └── synthesizer   [generation]  model + token + cost auto-tracked
```

Key methods used:
- `langfuse.update_current_generation(input, output, model, usage_details, metadata)`
- `langfuse.update_current_span(name, input, output, level, status_message)`
- `langfuse.set_current_trace_io(input, output)`
- `propagate_attributes(tags=[...])` — attaches `["pipeline", "mmm"]` tags to all child spans

Chart base64 blobs are intentionally excluded from Langfuse output to avoid bloating the trace.

---

## Data Flow — Vector Store

```
mmm_dummy_data/*.json
        │
   embedder.py (chunk_mmm_file → ingest_all)
        │
   ChromaDB PersistentClient  (./mmm_vectorstore)
        │
   agentic_retriever  (query with where filters)
        │
   pipeline synthesizer
```

---

## Environment Variables Required

| Variable | Purpose |
|---|---|
| `LANGFUSE_PUBLIC_KEY` | Langfuse project public key |
| `LANGFUSE_SECRET_KEY` | Langfuse project secret key |
| `GEMINI_API_KEY` | Google Gemini API key (via litellm) |
