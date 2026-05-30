"""
Edge-routing predicates for the MMM LangGraph.

Each function is a pure predicate — it reads state and returns a routing key.
Keeping routing logic here instead of inside nodes makes the graph topology
explicit and makes conditional edges unit-testable without running the graph.
"""

from __future__ import annotations

import logging
from typing import Literal

from .state import MMMGraphState

logger = logging.getLogger("mmm.graph.routing")

# ── Node name constants ────────────────────────────────────────────────────────
# Using constants avoids silent typos in conditional_edges dicts.

EXECUTOR = "executor"
REJECTED = "rejected"
SYNTHESIZER = "synthesizer"
END = "__end__"


# ── Routing predicates ─────────────────────────────────────────────────────────

def route_after_planner(
    state: MMMGraphState,
) -> Literal["human_approval", "__end__"]:
    """
    After the planner runs:
    - If the planner wrote an error, abort immediately.
    - Otherwise proceed to human review.

    This gate is only wired into the graph when use_human_approval=True.
    """
    if state.get("error") or state.get("plan") is None:
        trace_id = state.get("trace_id", "-")
        logger.warning("[%s] route_after_planner: plan error — aborting", trace_id)
        return END
    return "human_approval"


def route_after_approval(
    state: MMMGraphState,
) -> Literal["executor", "rejected"]:
    """
    After human approval:
    - 'approve' (case-insensitive) → executor
    - anything else → rejected
    """
    trace_id = state.get("trace_id", "-")
    feedback = (state.get("human_feedback") or "").strip().lower()
    decision = EXECUTOR if feedback == "approve" else REJECTED
    logger.info("[%s] route_after_approval: feedback=%r  decision=%s", trace_id, feedback, decision)
    return decision  # type: ignore[return-value]


def route_after_executor(
    state: MMMGraphState,
) -> Literal["synthesizer", "__end__"]:
    """
    After execution:
    - If executor wrote a hard error AND produced no results, abort.
    - Otherwise synthesize (partial results are still worth summarising).
    """
    trace_id = state.get("trace_id", "-")
    has_results = bool(state.get("execution_log"))
    has_error = bool(state.get("error"))

    if has_error and not has_results:
        logger.warning("[%s] route_after_executor: no results + error — aborting", trace_id)
        return END

    return SYNTHESIZER


def route_after_planner_no_approval(
    state: MMMGraphState,
) -> Literal["executor", "__end__"]:
    """
    Variant used when human approval is disabled.
    Planner success → executor; planner error → end.
    """
    if state.get("error") or state.get("plan") is None:
        trace_id = state.get("trace_id", "-")
        logger.warning("[%s] route_after_planner_no_approval: plan error — aborting", trace_id)
        return END
    return EXECUTOR
