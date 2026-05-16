# pipeline.py — Walkthrough

## Overview

`pipeline.py` is the core reasoning engine. It takes a plain English question and returns a structured answer with an optional chart. Every call makes exactly two LLM requests and runs one pure-Python execution loop in between.

```
Question
   │
   ▼
run_planner()        ← LLM call #1
   │  TaskPlan
   ▼
run_executor()       ← no LLM, pure Python tool calls
   │  execution_log
   ▼
run_synthesizer()    ← LLM call #2
   │
   ▼
CopilotAnswer dict   (summary, insights, narrative, chart_b64, sources)
```

---

## Data Models

Defined at the top of the file using Pydantic. These are the contracts between stages.

### `SubTask`

Represents one tool call that the planner wants to make.

```python
class SubTask(BaseModel):
    task:      str   # plain English description of what this step does
    tool_name: str   # must match a key in tool_registry
    tool_args: dict  # passed verbatim to the tool function
```

### `TaskPlan`

The full plan returned by the planner. A list of `SubTask` objects plus metadata.

```python
class TaskPlan(BaseModel):
    objective: str        # one-sentence goal
    subtasks:  list[SubTask]
    reasoning: str        # why these tools in this order
```

### `ChannelInsight`

One row in the per-channel breakdown table in the final answer.

```python
class ChannelInsight(BaseModel):
    channel:        str
    current_roi:    float
    benchmark_roi:  Optional[float]
    recommendation: str
    confidence:     str   # "high" | "medium" | "low"
```

### `CopilotAnswer`

The full structured response from the synthesizer.

```python
class CopilotAnswer(BaseModel):
    summary:    str                   # 2-3 sentence executive summary
    insights:   list[ChannelInsight]  # per-channel ROI comparison rows
    narrative:  str                   # 3-5 paragraph written analysis
    chart_type: Optional[str]         # which chart was generated
    sources:    list[str]             # brand names referenced in the answer
```

> `chart_b64` is added to the returned dict after Pydantic parsing — it is the base64 PNG string extracted from the executor log. It is not declared in `CopilotAnswer` because it bypasses the LLM context entirely.

---

## Tool Registry

### `build_tool_registry(collection) → dict`

Called once per pipeline run. Returns a dict mapping tool name strings to callable Python functions. Each inner function is a closure that captures `collection` and normalises the raw `args` dict before delegating to the actual module-level function.

```
"mmm_retriever"              → _mmm_retriever(args)
"benchmark_fetcher"          → _benchmark_fetcher(args)
"scenario_simulator"         → _scenario_simulator(args)
"roi_bar_chart"              → _roi_bar_chart(args)
"spend_vs_revenue_chart"     → _spend_vs_revenue_chart(args)
"revenue_waterfall_chart"    → _revenue_waterfall_chart(args)
"scenario_bar_chart"         → _scenario_bar_chart(args)
"benchmark_comparison_chart" → _benchmark_comparison_chart(args)
"power_curve_chart"          → _power_curve_chart(args)
```

Each wrapper handles optional args with `.get()` defaults so the LLM does not need to supply every field. For example, `_benchmark_fetcher` checks whether `args` contains `"channels"` (batch mode) or `"channel"` (single mode) and routes accordingly.

---

## Stage 1 — Planner

### `run_planner(question) → TaskPlan`

**What it does**: sends the question to the LLM with a system prompt that describes all 9 tools, their argument schemas, and planning rules. The LLM returns a JSON object; this function parses it into a `TaskPlan`.

**System prompt rules enforced**:
1. Always start with `mmm_retriever` to ground the plan in real data
2. Only include `scenario_simulator` if the question involves budget changes or spend shifts
3. Only include `benchmark_fetcher` if the question asks about industry comparison or benchmarks
4. End with exactly one chart tool that best visualises the answer
5. Return only valid JSON — no markdown, no commentary

**Parsing**:
- Strips markdown code fences (` ```json `) from the response
- Parses JSON and validates against `TaskPlan`
- On any parse failure falls back to a minimal single-step plan: `mmm_retriever` only

**Logging**: logs the objective, number of subtasks, reasoning summary, and each step with its tool name at `INFO` level.

---

## Stage 2 — Executor

### `run_executor(plan, tool_registry) → list[dict]`

**What it does**: iterates over `plan.subtasks` in order and calls each tool. No LLM involved.

**Per-subtask loop**:
1. Looks up `subtask.tool_name` in `tool_registry`
2. If the tool is not registered, appends an error entry and continues (does not raise)
3. Calls `tool_fn(subtask.tool_args)` and captures the return value
4. Appends a log entry: `{ task, tool_name, result }`

**Error handling**: each tool call is wrapped in a `try/except`. Errors are recorded as `"ERROR: <message>"` strings in the log entry. This allows the synthesizer to acknowledge missing data gracefully rather than crashing the pipeline.

**Result preview logging**: results longer than 120 characters are truncated for the log line to avoid flooding logs with base64 chart blobs.

**Return value**: a list of dicts — one per executed subtask:

```python
[
    { "task": "Retrieve relevant MMM data", "tool_name": "mmm_retriever",    "result": "--- Result 1 ---\n..." },
    { "task": "Compare to benchmarks",      "tool_name": "benchmark_fetcher", "result": { "status": "found", ... } },
    { "task": "Generate ROI chart",         "tool_name": "roi_bar_chart",     "result": "<base64 PNG string>" },
]
```

---

## Stage 3 — Synthesizer

### `run_synthesizer(question, execution_log) → dict`

**What it does**: builds a context string from all tool results, sends it to the LLM, and parses the structured JSON answer.

**Context building**:
- Iterates over `execution_log`
- Chart tools are detected by checking `"chart" in tool_name` and whether the result is a long string (`> 200` chars). Their base64 content is **extracted and stored separately** as `chart_b64` — it is never sent to the LLM
- For all other tools, the result is serialised to JSON (dicts) or string, truncated at 3000 characters to stay within context limits
- Sections are joined with `---` separators

**LLM call**: sends the context bundle to the synthesizer system prompt, which instructs the model to return a `CopilotAnswer`-shaped JSON object. The prompt explicitly forbids made-up numbers — all figures must come from tool results.

**Parsing**: strips code fences, parses JSON. On failure, falls back to a minimal answer with the raw LLM text as the `narrative` and empty `insights`.

**Return value**: the parsed answer dict with `chart_b64` injected at the top level:

```python
{
    "summary":    "...",
    "insights":   [...],
    "narrative":  "...",
    "chart_type": "roi_bar",
    "sources":    ["keytruda", "lenvima"],
    "chart_b64":  "<base64 PNG>",   # injected after parsing, not from LLM
}
```

---

## Entry Point

### `run(question, collection) → dict`

The public function called by `app.py` (Gradio) and `main.py` (FastAPI). Orchestrates the three stages in order:

```python
tool_registry = build_tool_registry(collection)   # wires up all tool closures
plan          = run_planner(question)              # LLM call #1
execution_log = run_executor(plan, tool_registry)  # pure Python
answer        = run_synthesizer(question, execution_log)  # LLM call #2
return answer
```

Logs a separator line and `"Pipeline start"` / `"Pipeline complete"` at `INFO` level for easy tracing across multi-question runs.

---

## Terminal Renderer

### `print_answer(answer) → None`

Used only when running `pipeline.py` directly from the command line. Formats the answer dict as readable console output: summary, channel insights table, narrative, and chart type. Not used by either the Gradio UI or FastAPI server.

---

## Flow Diagram (with data types)

```
question: str
    │
    ▼
build_tool_registry(collection)
    │  dict[str, Callable[[dict], Any]]
    │
    ▼
run_planner(question)
    │  LiteLLM → raw JSON string → TaskPlan
    │
    ▼
run_executor(plan, tool_registry)
    │  list of subtasks → list[{ task, tool_name, result }]
    │
    ├── mmm_retriever    → str   (formatted retrieval results)
    ├── benchmark_fetcher → dict  (ROI comparison vs benchmarks)
    ├── scenario_simulator → dict (before/after revenue projections)
    └── *_chart          → str   (base64 PNG, extracted by synthesizer)
    │
    ▼
run_synthesizer(question, execution_log)
    │  LiteLLM → raw JSON string → dict + chart_b64 injected
    │
    ▼
dict: { summary, insights[], narrative, chart_type, sources, chart_b64 }
```
