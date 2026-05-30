"""
State schema and configuration for the MMM LangGraph pipeline.

Design principles
-----------------
- Each field carries an explicit reducer so concurrent node writes are deterministic.
- Default (no reducer) is last-write-wins — only deviate where append/accumulate semantics
  are needed (execution_log, tools_used, retry_count).
- GraphConfig centralises every tunable parameter so callers never pass loose kwargs.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Optional

from typing_extensions import TypedDict


# ── Custom reducers ────────────────────────────────────────────────────────────

def _append_list(current: list | None, update: list | None) -> list:
    """Append new items to an existing list; treat None as empty list."""
    return (current or []) + (update or [])


# ── State ──────────────────────────────────────────────────────────────────────

class MMMGraphState(TypedDict):
    # Immutable inputs — set once at entry, never overwritten by nodes
    question: str
    trace_id: str  # UUID correlation ID for structured logging / Langfuse

    # Planner output — last-write-wins (default)
    plan: Optional[dict]

    # Human reviewer decision — set by interrupt resume
    human_feedback: Optional[str]

    # Execution results — each executor invocation appends to the list
    execution_log: Annotated[list, _append_list]
    tools_used: Annotated[list, _append_list]

    # Synthesizer output — last-write-wins
    answer: Optional[dict]

    # Error message from any node — last-write-wins
    error: Optional[str]

    # Increments by 1 on each retry attempt
    retry_count: Annotated[int, operator.add]


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class GraphConfig:
    """All tunable graph parameters in a single, injectable dataclass."""

    checkpointer_path: str = "mmm_checkpoints.db"
    max_retries: int = 2
    use_human_approval: bool = True
    # Future: model_override, temperature, tool_timeout_seconds, etc.
    extra: dict = field(default_factory=dict)
