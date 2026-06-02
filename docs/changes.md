# Changes Log

## Build 3 — Current (Bug Fixes + Observability Polish)

### Bug Fixes

**`src/agentic_retriver.py` — `build_where_clause` missing `question` argument**
- `build_where_clause(filters, question)` requires two arguments; the call site passed only `filters`
- This caused a `TypeError` on every retrieval attempt that had non-empty filters
- Fix: pass `question` as second argument so `needs_summary_chunk` can check whether to add `{"type": "summary"}` to the where clause

**`src/agentic_retriver.py` — `agentic_retriever` did not accept `n_results`**
- `pipeline.py` called `mmm_retriever(question=..., collection=..., n_results=...)` via the tool registry wrapper
- `agentic_retriever()` had no `n_results` parameter, causing `TypeError: unexpected keyword argument 'n_results'`
- Fix: added `n_results: int = 10` parameter; ChromaDB queries now use the caller-supplied value instead of the hardcoded `10`

**`src/chart_generator.py` — `benchmark_comparison_chart` crashed on `None` ROI**
- `benchmark_fetcher` deliberately returns `current_roi=None` when no brand ROI is available (the synthesizer is expected to extract it from MMM retrieval results)
- The chart sorted the result list by `x["current_roi"]` — this raised `TypeError: '<' not supported between NoneType and float`
- Fix: filter to only entries with non-`None` `current_roi` before sorting; returns empty string (no chart) if none qualify

---

## Build 2 — LangGraph + Langfuse Observability (`feat: add FAANG-standard LangGraph pipeline`)

### What was added

**Langfuse v4 tracing in `pipeline.py`**
- Wrapped `run_planner`, `run_executor`, `_run_single_tool`, `run_synthesizer`, and `run` with `@observe`
- Planner and synthesizer use `as_type="generation"` for automatic token + cost tracking
- Each tool call creates a named child span with `input`, `output`, `level`, and `status_message`
- Chart base64 blobs are excluded from Langfuse output to avoid trace bloat
- `langfuse.flush()` added to `__main__` for short-lived script safety
- Why: tracing was absent in the initial build; every LLM call and tool execution was a black box

**`src/graph/` — LangGraph pipeline**
- Added human-in-the-loop approval via `interrupt()` before execution begins
- `SqliteSaver` checkpointer persists state across process restarts
- `build_graph_with_approval()` / `build_graph_silent()` factory pattern keeps graph topology explicit
- Routing predicates in `routing.py` are pure functions, unit-testable without running the graph
- `run_with_approval()` and `run_silent()` provide clean public entry points
- Why: the flat `pipeline.py` had no mechanism for plan review or recovery from mid-run failures; LangGraph provides both via `interrupt` + checkpointing

**`src/agentic_retriver.py` — Iterative RAG retriever**
- Replaced the commented-out single-shot `mmm_retriever` with a 3-step loop: query rewrite → keyword check → LLM-as-judge
- Gap-aware re-query on retry: the LLM judge's gap description is fed back into the next query rewrite
- `did_you_mean` fuzzy-match suggestion for unknown brand/channel names
- Why: single-shot retrieval frequently returned irrelevant chunks when brand names or channels were phrased differently from the stored metadata

---

## Build 1 — Initial Commit (`Initial commit: MMM Copilot with Gradio UI`)

### What was built

- `src/config.py`, `src/db.py`, `src/embedder.py` — ChromaDB vector store with sentence-transformer embeddings
- `src/generate_mmm_output.py` — synthetic MMM JSON data generator for 19 pharma brands
- `src/query_parser.py` — regex + LLM filter extraction for ChromaDB `where` clauses
- `src/benchmark_fetcher.py` — static pharma ROI benchmark table (14 channels × 3 verticals)
- `src/scenario_simulator.py` — power-curve budget simulation
- `src/chart_generator.py` — 6 Matplotlib chart types returned as base64 PNG
- `pipeline.py` — planner → executor → synthesizer pipeline with litellm LLM calls
- `autoanalyst.ipynb` — Gradio UI and notebook runner

### Limitations in Build 1
- No observability — all LLM calls were untraced
- Single-shot retrieval — no retry or relevance check
- No human review — plans executed immediately without approval option
- `build_where_clause` accepted `question` parameter but the agentic retriever (added in Build 2) called it without it
