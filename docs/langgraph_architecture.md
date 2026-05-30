# LangGraph Architecture — MMM Copilot

## Overview

The `src/graph/` module wraps the existing three-stage MMM pipeline
(planner → executor → synthesizer) in a stateful, checkpointed LangGraph
that adds human-in-the-loop approval and clean error paths.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      src/graph (LangGraph layer)                    │
│                                                                     │
│  ┌────────────┐   ┌──────────────────┐   ┌──────────────────────┐  │
│  │   state.py │   │    nodes.py       │   │     routing.py       │  │
│  │            │   │                  │   │                      │  │
│  │ MMMGraph   │   │ planner_node     │   │ route_after_planner  │  │
│  │   State    │◀──│ human_approval   │   │ route_after_approval │  │
│  │            │   │ executor_node    │   │ route_after_executor │  │
│  │ GraphConfig│   │ synthesizer_node │   │                      │  │
│  │            │   │ rejected_node    │   │                      │  │
│  └────────────┘   └──────────────────┘   └──────────────────────┘  │
│                            │                        │               │
│                            ▼                        ▼               │
│                   ┌──────────────────┐   ┌──────────────────────┐  │
│                   │    graph.py      │   │      runner.py       │  │
│                   │                 │   │                      │  │
│                   │ build_graph_    │   │ run_with_approval()  │  │
│                   │   with_approval │   │ run_silent()         │  │
│                   │ build_graph_    │   │                      │  │
│                   │   silent        │   │                      │  │
│                   └──────────────────┘   └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                            │
                            ▼ delegates to
┌─────────────────────────────────────────────────────────────────────┐
│                        pipeline.py (existing)                       │
│                                                                     │
│   run_planner()        run_executor()        run_synthesizer()      │
│   (Langfuse-traced)    (Langfuse-traced)     (Langfuse-traced)      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Layer responsibilities

### `state.py` — Single source of truth for state

`MMMGraphState` is a `TypedDict` carrying all data that flows between nodes.
Each field has an explicit **reducer** — the function LangGraph calls when two
nodes write to the same field concurrently:

```
field           reducer         rationale
──────────────  ──────────────  ──────────────────────────────────────────
question        last-write      set once at entry; never mutated by nodes
trace_id        last-write      set once; UUID for log correlation
plan            last-write      planner always overwrites the previous plan
human_feedback  last-write      interrupt resume always provides fresh value
execution_log   _append_list    accumulates tool results; safe across retries
tools_used      _append_list    accumulates; safe across retries
answer          last-write      synthesizer always produces a final answer
error           last-write      last error wins; earlier errors are superseded
retry_count     operator.add    increments by 1 per retry
```

`GraphConfig` is a `dataclass` (not part of the graph state) injected by the
caller to control checkpointer path, retry limits, and approval toggle.

### `nodes.py` — Business logic

Each function signature is `(MMMGraphState) -> dict[str, Any]`.  Nodes only
return the fields they write — LangGraph applies the partial update using
the reducers above.  This means nodes are decoupled from each other's
concerns and trivially unit-testable.

Nodes delegate all LLM and tool calls to `pipeline.py` — the graph layer
adds no new AI logic, only orchestration.

### `routing.py` — Explicit conditional logic

All `if/else` routing decisions are pure functions that take state and return
a routing key.  Extracting them from the graph definition makes the control
flow human-readable and independently testable:

```python
def route_after_approval(state: MMMGraphState) -> Literal["executor", "rejected"]:
    feedback = (state.get("human_feedback") or "").strip().lower()
    return "executor" if feedback == "approve" else "rejected"
```

### `graph.py` — Topology declaration

Graph assembly is separated from graph execution.  The two factory functions
(`build_graph_with_approval`, `build_graph_silent`) return uncompiled
`StateGraph` objects — the checkpointer is injected at compile time by
`runner.py`.  This makes it possible to compile the same graph topology with
different checkpointers (MemorySaver for tests, SqliteSaver for production).

### `runner.py` — Entry points and I/O

Runner functions own:
- Thread ID generation (one UUID per run, enabling parallel runs + replay)
- Checkpointer lifecycle (context manager — always closed cleanly)
- Phase 1 / Phase 2 split for the interrupt/resume pattern
- Stdout rendering and structured logging

---

## Interrupt / resume lifecycle

```
Thread 1: run_with_approval() ──────────────────────────────────────────────▶
                                │                              │
         app.invoke(initial)    │                              │
                                ▼                              │
                          [ planner_node ]                     │
                                │                              │
                          [ human_approval_node ]              │
                              interrupt() ◀── pauses here      │
                                │                              │
                         returns { "__interrupt__": [payload] }│
                                                               │
         human reads plan + types "approve"                    │
                                                               │
         app.invoke(Command(resume="approve"), config)         │
                                                               ▼
                                                     [ executor_node ]
                                                     [ synthesizer_node ]
                                                     returns final state
```

The `thread_id` in `run_config` is what links Phase 1 and Phase 2 — LangGraph
uses it to look up the saved checkpoint and replay from the interruption point.

---

## Checkpointing and persistence

| Checkpointer | When to use |
|---|---|
| `SqliteSaver` | Production and interactive CLI — survives process restarts |
| `MemorySaver` | Tests and notebooks — no I/O overhead |

The SQLite file path is controlled by `GraphConfig.checkpointer_path`.
Each run gets its own `thread_id` so checkpoint histories never collide.

---

## Error paths

```
planner fails           → error written to state → route_after_planner → END
human rejects plan      → rejected_node → error written → END
executor fatal failure  → error written, no execution_log → route_after_executor → END
executor partial failure → error written, but execution_log populated
                         → synthesizer still runs (partial results > no answer)
synthesizer fails       → error written to state → END
```

Nodes never raise — they catch all exceptions and encode them as
`{"error": "..."}` partial state updates.  The routing predicates then
decide whether to short-circuit or continue.

---

## Observability

Every node logs structured lines keyed by `trace_id`.  A single `grep` or
log query on the trace_id returns the full node sequence for any run:

```
grep "a1b2c3d4" app.log

[a1b2c3d4] planner: generating plan  question='...'
[a1b2c3d4] planner: done  objective='...'  subtasks=4
[a1b2c3d4] human_approval: pausing for review  steps=4
[a1b2c3d4] human_approval: resumed  feedback='approve'
[a1b2c3d4] executor: starting  ...
[a1b2c3d4] executor: done  tools=4  errors=0
[a1b2c3d4] synthesizer: done  insights=3  chart=roi_bar
```

The `trace_id` is forwarded into Langfuse via the existing `@observe`
instrumentation in `pipeline.py`, giving end-to-end visibility from the
LangGraph run down to individual LLM calls and tool spans.

---

## Design decisions

**Why split into six files?**
Each file has exactly one reason to change.  Adding a new node only touches
`nodes.py` and `graph.py`.  Changing routing logic only touches `routing.py`.
A single-file graph (like `mmm_graph.py`) makes all of these concerns collide.

**Why TypedDict over Pydantic for state?**
LangGraph requires `TypedDict` (or `dataclass`) for state — Pydantic models
are not directly supported.  `MMMGraphState` carries the wire format;
`TaskPlan` / `CopilotAnswer` in `pipeline.py` provide the richer validation.

**Why delegate to pipeline.py instead of reimplementing?**
The existing pipeline.py already has battle-tested Langfuse instrumentation,
prompt engineering, and fallback logic.  The graph layer adds orchestration;
it does not duplicate AI logic.

**Why two graph variants instead of a flag inside one graph?**
An interrupt node in a graph always requires a persistent checkpointer.
A graph without an interrupt can use an in-memory saver with lower overhead.
Having two clean graph definitions is simpler than one graph that conditionally
includes interrupt nodes.

**Why not use LangGraph's built-in retry?**
LangGraph's `RetryPolicy` retries on exception.  Our nodes never raise — they
return error state instead.  This was a deliberate choice: it keeps routing
logic in `routing.py` (observable, testable) rather than inside exception
handlers (opaque).  A future retry subgraph can be added by extending
`routing.py` and wiring new edges in `graph.py`.
