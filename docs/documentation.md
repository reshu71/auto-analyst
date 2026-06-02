# MMM Copilot — Documentation

## Setup

### 1. Install dependencies

```bash
uv sync
# or
pip install -e .
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
GEMINI_API_KEY=AIza...
```

### 3. Generate and ingest data

```bash
# Generate synthetic MMM JSON files into mmm_dummy_data/
python -m src.generate_mmm_output

# Ingest into ChromaDB vector store
python -m src.embedder
```

### 4. Run the pipeline

**Direct (CLI):**
```bash
python pipeline.py
```

**LangGraph — silent (no human approval):**
```bash
python -m src.graph.runner "What is the ROI of HCP channels for Keytruda?"
```

**LangGraph — with human approval prompt:**
```python
from src.graph.runner import run_with_approval
result = run_with_approval("Which channels are underperforming for Keytruda?")
```

---

## Pipeline API

### `pipeline.run(question, collection) → dict`

Main entry point. Returns a `CopilotAnswer`-shaped dict:

```python
{
    "summary":    str,          # 2-3 sentence executive summary
    "insights":   list[dict],   # per-channel ROI + benchmark + recommendation
    "narrative":  str,          # 3-5 paragraph detailed analysis
    "chart_type": str,          # "roi_bar" | "waterfall" | "scenario" | "benchmark" | "scatter" | "power_curve" | "none"
    "sources":    list[str],    # brand/model names referenced
    "tools_used": list[str],    # tool names that ran
    "chart_b64":  str | None,   # base64 PNG — ready for <img src="data:image/png;base64,...">
}
```

### `pipeline.run_planner(question) → TaskPlan`

Calls the LLM planner. Returns a structured plan with objective, reasoning, and ordered subtasks. Traced as a Langfuse generation.

### `pipeline.run_executor(plan, tool_registry) → list[dict]`

Executes each subtask in order. Returns an execution log: `[{"task", "tool_name", "result"}, ...]`. Each tool runs in its own Langfuse span.

### `pipeline.run_synthesizer(question, execution_log) → dict`

Synthesizes tool results into a `CopilotAnswer` dict. Traced as a Langfuse generation.

---

## Tool Registry

Tools are registered per-collection in `build_tool_registry(collection)`. The planner selects tools by name; the executor dispatches via the registry.

### `mmm_retriever`

Calls the agentic iterative retriever.

```python
args: { "question": str, "n_results": int }  # n_results defaults to 8
```

### `benchmark_fetcher`

Fetches industry ROI benchmarks.

```python
# Single channel
args: { "channel": str, "sub_vertical": str }

# Batch
args: { "channels": [{"name": str}], "sub_vertical": str }
```

Sub-vertical must be one of: `"oncology"`, `"vaccines"`, `"pharma"`.

### `scenario_simulator`

Simulates budget change impact on revenue using a power-curve model.

```python
# Single channel
args: {
    "channel": str,
    "current_spend": float,
    "current_revenue_contribution": float,
    "budget_change_pct": float  # -0.20 = cut 20%, +0.15 = increase 15%
}

# Portfolio
args: {
    "channels": [{"name": str, "spend": float, "revenue_contribution": float}],
    "budget_changes": {"channel_name": float}
}
```

### Chart tools

All chart tools return a base64 PNG string.

| Tool | Required args |
|---|---|
| `roi_bar_chart` | `channels: list[{name, roi, category}]`, `brand`, `sub_vertical` |
| `spend_vs_revenue_chart` | `channels: list[{name, spend, revenue_contribution, roi}]`, `brand` |
| `revenue_waterfall_chart` | `model_summary: {total_revenue, base_contribution_pct}`, `channels`, `brand` |
| `scenario_bar_chart` | `scenario_results: list` (output of scenario_simulator) |
| `benchmark_comparison_chart` | `benchmark_results: list` (output of benchmark_fetcher with current_roi set), `sub_vertical` |
| `power_curve_chart` | `channel: str`, `current_spend: float`, `current_revenue: float` |

---

## Agentic Retriever

`agentic_retriever(question, collection, n_results=10, max_retries=3)`

The retriever runs a loop of up to `max_retries` attempts:

1. **Query rewrite** — LLM expands the question into a dense keyword string for semantic similarity. On retry, the rewrite targets the gap identified in the previous attempt.
2. **ChromaDB query** — applies brand/year/channel/vertical filters extracted by `query_parser`; falls back to unfiltered search if no filters apply.
3. **Zero-result check** — if ChromaDB returns nothing, returns a `did_you_mean` suggestion using fuzzy matching against known entity names.
4. **Keyword check** — fast check that retrieved chunks mention expected brands/channels/year. If <50% checks pass, retries.
5. **LLM-as-judge** — deep sufficiency audit. If sufficient, returns results. If not, updates the gap description and retries.

Returns the best results seen across all attempts, or an informative error string.

---

## LangGraph Runner

### Silent mode (no approval)

```python
from src.graph.runner import run_silent
from src.graph.state import GraphConfig

result = run_silent(
    "What is the ROI of HCP channels for oncology brands?",
    config=GraphConfig(checkpointer_path="mmm_checkpoints.db")
)
print(result["answer"]["summary"])
```

### Human-in-the-loop mode

```python
from src.graph.runner import run_with_approval

# Pauses at interrupt, prints the plan, prompts for 'approve' or rejection feedback
result = run_with_approval("Which channels are underperforming for Keytruda?")
```

### `GraphConfig`

```python
@dataclass
class GraphConfig:
    checkpointer_path: str = "mmm_checkpoints.db"  # SQLite path for state persistence
    max_retries: int = 2
    use_human_approval: bool = True
```

### State shape (`MMMGraphState`)

| Field | Type | Reducer |
|---|---|---|
| `question` | `str` | last-write-wins |
| `trace_id` | `str` | last-write-wins |
| `plan` | `dict \| None` | last-write-wins |
| `human_feedback` | `str \| None` | last-write-wins |
| `execution_log` | `list` | append |
| `tools_used` | `list` | append |
| `answer` | `dict \| None` | last-write-wins |
| `error` | `str \| None` | last-write-wins |
| `retry_count` | `int` | add |

---

## Langfuse Observability

### Trace structure

Every `pipeline.run()` call creates one trace named `mmm_pipeline` containing:

```
mmm_pipeline  [trace]
  ├── planner       [generation]
  │     input:  system prompt + user question
  │     output: raw JSON plan
  │     model:  gemini/...
  │     usage:  { input, output, total } tokens
  │     metadata: { objective, reasoning, num_subtasks, tools_planned }
  │
  ├── executor      [span]
  │     input:  { objective, subtasks }
  │     output: { tools_executed, num_results, errors }
  │     level:  WARNING if any tool errored
  │     │
  │     ├── <tool_name>  [span]   (one per subtask)
  │     │     input:  tool_args dict
  │     │     output: result (truncated at 500 chars; chart blobs replaced with placeholder)
  │     │     level:  ERROR on failure
  │
  └── synthesizer   [generation]
        input:  system prompt + tool context
        output: raw JSON answer
        model:  gemini/...
        usage:  { input, output, total } tokens
        metadata: { num_insights, chart_type, sources, has_chart }
```

### Tags

All spans under a pipeline run are tagged `["pipeline", "mmm"]` via `propagate_attributes`.

### Flushing

In short-lived scripts, call `langfuse.flush()` before exit to ensure all events are sent:

```python
from langfuse import get_client
langfuse = get_client()
# ... run pipeline ...
langfuse.flush()
```

---

## Benchmark Data

The `src/benchmark_fetcher.py` table covers 14 channels across 3 pharma sub-verticals.

| Channel | Category | Notes |
|---|---|---|
| doximity | HCP digital | High ROI, broad HCP reach |
| medscape | HCP digital | Strong for oncology |
| pulsepoint | HCP digital | Display-focused, lower ROI |
| deepintent | HCP digital | Intent-targeted display |
| sfmc | HCP digital | Email; engagement benchmarks available |
| nexgen | HCP digital | Email |
| salesforce_calls | HCP | Highest absolute revenue, lowest efficiency ROI |
| tv | Consumer | Low ROI for oncology; high for vaccines |
| streaming_tv | Consumer | Better targeting than linear |
| online_video | Consumer | Mid-range |
| paid_search | Consumer | Highest ROI, near-linear response curve |
| social | Consumer | Facebook/Instagram |
| display | Consumer | Lowest ROI across all channels |
| audio | Consumer | Lowest ROI alongside display |

Source: aggregated from 100+ pharma MMM studies, 80+ brands (2019–2024).

---

## Project Structure

```
auto-analyst/
├── pipeline.py                 # Core planner → executor → synthesizer
├── pyproject.toml
├── .env                        # LANGFUSE_* and GEMINI_API_KEY (not committed)
├── mmm_vectorstore/            # ChromaDB persistent store
├── mmm_dummy_data/             # Synthetic MMM JSON files
├── mmm_checkpoints.db          # LangGraph SQLite checkpointer
├── src/
│   ├── config.py               # Global constants (model, paths, year)
│   ├── db.py                   # ChromaDB collection factory
│   ├── embedder.py             # JSON → chunks → ChromaDB ingestion
│   ├── generate_mmm_output.py  # Synthetic data generator
│   ├── query_parser.py         # Filter extraction (regex + LLM fallback)
│   ├── agentic_retriver.py     # Iterative RAG retriever
│   ├── benchmark_fetcher.py    # Static pharma ROI benchmark table
│   ├── scenario_simulator.py   # Power-curve budget simulation
│   ├── chart_generator.py      # 6 Matplotlib chart types → base64 PNG
│   └── graph/
│       ├── state.py            # MMMGraphState TypedDict + GraphConfig
│       ├── nodes.py            # Node functions (planner, executor, synthesizer, approval)
│       ├── routing.py          # Conditional edge predicates
│       ├── graph.py            # Graph factory functions
│       └── runner.py           # run_with_approval / run_silent entry points
└── docs/
    ├── architecture.md
    ├── changes.md
    └── documentation.md
```
