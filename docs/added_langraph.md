# Added: LangGraph Integration

## What was added

A production-quality LangGraph pipeline that wraps the existing `pipeline.py` planner / executor / synthesizer stages in a stateful, checkpointed graph with a human-in-the-loop approval step.

**New files (nothing existing was modified):**

```
src/graph/
├── __init__.py      — public API
├── state.py         — MMMGraphState TypedDict + GraphConfig dataclass
├── nodes.py         — node implementations (planner, human_approval, executor, synthesizer, rejected)
├── routing.py       — pure routing predicates (conditional edges)
├── graph.py         — graph assembly (with-approval + silent variants)
└── runner.py        — run_with_approval() and run_silent() entry points
```

---

## Quick start

### Interactive run (human approval)

```python
from src.graph import run_with_approval, GraphConfig

result = run_with_approval(
    "What is the ROI of HCP channels for oncology brands?",
    config=GraphConfig(checkpointer_path="mmm_checkpoints.db"),
)

if result.get("answer"):
    from pipeline import print_answer
    print_answer(result["answer"])
```

The graph pauses after planning and prints the proposed tool steps.
Type **`approve`** to proceed or any other text to reject with feedback.

### Programmatic / batch run (no approval)

```python
from src.graph import run_silent, GraphConfig

result = run_silent(
    "Which channels are underperforming for Keytruda?",
    config=GraphConfig(checkpointer_path="mmm_checkpoints.db"),
)
```

### From the command line

```bash
python -m src.graph.runner "What is the ROI of HCP channels for oncology brands?"
```

---

## Configuration

All tunable parameters live in `GraphConfig`:

| Field | Default | Description |
|---|---|---|
| `checkpointer_path` | `"mmm_checkpoints.db"` | SQLite file for LangGraph checkpoints |
| `max_retries` | `2` | Reserved for future retry subgraph |
| `use_human_approval` | `True` | Toggle approval step (overridden to `False` by `run_silent`) |
| `extra` | `{}` | Pass-through dict for forward-compatibility |

---

## State fields

`MMMGraphState` flows through every node as a shared slate:

| Field | Reducer | Set by |
|---|---|---|
| `question` | last-write (default) | Entry point — never mutated |
| `trace_id` | last-write (default) | Entry point — UUID for log correlation |
| `plan` | last-write (default) | `planner_node` |
| `human_feedback` | last-write (default) | `human_approval_node` (interrupt resume) |
| `execution_log` | **append** | `executor_node` — accumulates across retries |
| `tools_used` | **append** | `executor_node` — accumulates across retries |
| `answer` | last-write (default) | `synthesizer_node` |
| `error` | last-write (default) | Any node on failure |
| `retry_count` | **add** | Reserved for retry subgraph |

---

## Graph topology

### With human approval (`run_with_approval`)

```
planner
  │
  ├─[error]──────────────────────────────────────────▶ END
  │
  ▼
human_approval  ◀── interrupt pauses here
  │
  ├─[approve]──▶ executor
  │                │
  │                ├─[no results + error]──────────────▶ END
  │                │
  │                ▼
  │             synthesizer ──────────────────────────▶ END
  │
  └─[reject]──▶ rejected ──────────────────────────────▶ END
```

### Silent / batch (`run_silent`)

```
planner
  │
  ├─[error]──────────────────────────────────────────▶ END
  │
  ▼
executor
  │
  ├─[no results + error]──────────────────────────────▶ END
  │
  ▼
synthesizer ──────────────────────────────────────────▶ END
```

---

## Observability

Every node emits structured log lines keyed by `trace_id`:

```
[a1b2c3d4] planner: generating plan  question='What is the ROI...'
[a1b2c3d4] planner: done  objective='...'  subtasks=4
[a1b2c3d4] human_approval: pausing for review  steps=4
[a1b2c3d4] human_approval: resumed  feedback='approve'
[a1b2c3d4] executor: starting  plan_objective='...'
[a1b2c3d4] executor: done  tools=4  errors=0
[a1b2c3d4] synthesizer: done  insights=3  chart=roi_bar
```

The `trace_id` is also forwarded to Langfuse through the existing `run_planner` /
`run_executor` / `run_synthesizer` instrumentation in `pipeline.py`, so every
LangGraph run is traceable end-to-end in the Langfuse dashboard.

---

## Relationship to mmm_graph.py

`mmm_graph.py` is the prototype that informed this implementation.
`src/graph/` is the production rewrite — it was built from scratch alongside
`mmm_graph.py` without modifying it.  Key differences:

| Concern | mmm_graph.py | src/graph/ |
|---|---|---|
| Module structure | Single file | Six files with single responsibilities |
| State reducers | No explicit reducers (last-write-wins only) | `Annotated` reducers for lists and counters |
| Routing | Inline lambda | Named functions in `routing.py` (unit-testable) |
| Error handling | Minimal | Every node catches exceptions, writes `error` to state |
| Graph variants | One (always with approval) | `with_approval` + `silent` |
| Entry point | Module-level code | `runner.py` functions |
| Config | Hard-coded strings | `GraphConfig` dataclass |
| Logging | `print()` | Structured `logging` with `trace_id` |
