"""
LangGraph node implementations for the MMM pipeline.

Each node is a pure function: (MMMGraphState) -> partial-state dict.
Nodes only write the fields they own — LangGraph merges partial updates
into the running state using the reducers declared in state.py.

Observability: every node emits structured log lines keyed by trace_id
so downstream log aggregators (Datadog, CloudWatch, etc.) can correlate
a full graph run from a single trace_id.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.types import interrupt

from pipeline import TaskPlan, build_tool_registry, run_executor, run_planner, run_synthesizer
from src.db import get_collection

from .state import MMMGraphState

logger = logging.getLogger("mmm.graph.nodes")


# ── Planner ────────────────────────────────────────────────────────────────────

def planner_node(state: MMMGraphState) -> dict[str, Any]:
    """
    Calls the LLM planner and writes the resulting TaskPlan dict to state.

    On failure, writes an error string and sets plan=None so routing can
    short-circuit to END rather than proceeding with a broken plan.
    """
    trace_id = state.get("trace_id", "-")
    question = state["question"]
    logger.info("[%s] planner: generating plan  question=%r", trace_id, question)

    try:
        plan_object = run_planner(question)
        plan_dict = plan_object.model_dump()
        logger.info(
            "[%s] planner: done  objective=%r  subtasks=%d",
            trace_id,
            plan_dict.get("objective", ""),
            len(plan_dict.get("subtasks", [])),
        )
        return {"plan": plan_dict}

    except Exception as exc:
        logger.error("[%s] planner: failed  error=%s", trace_id, exc, exc_info=True)
        return {"plan": None, "error": f"Planner failed: {exc}"}


# ── Human approval (interrupt) ─────────────────────────────────────────────────

def human_approval_node(state: MMMGraphState) -> dict[str, Any]:
    """
    Pauses execution and surfaces the plan for human review via LangGraph interrupt.

    The graph will resume when the caller invokes:
        app.invoke(Command(resume=<feedback>), config=run_config)

    The interrupt payload is structured so a UI can render it without parsing
    free-form text — all fields are named and typed.
    """
    trace_id = state.get("trace_id", "-")
    proposed_plan = state["plan"] or {}
    steps = [
        f"Step {i + 1}: {s['task']}"
        for i, s in enumerate(proposed_plan.get("subtasks", []))
    ]
    logger.info("[%s] human_approval: pausing for review  steps=%d", trace_id, len(steps))

    feedback = interrupt(
        {
            "trace_id": trace_id,
            "question": state["question"],
            "objective": proposed_plan.get("objective", ""),
            "reasoning": proposed_plan.get("reasoning", ""),
            "steps": steps,
            "message": (
                "Review the plan above. "
                "Type 'approve' to proceed or provide feedback to cancel."
            ),
        }
    )

    logger.info("[%s] human_approval: resumed  feedback=%r", trace_id, feedback)
    return {"human_feedback": feedback}


# ── Rejection ──────────────────────────────────────────────────────────────────

def rejected_node(state: MMMGraphState) -> dict[str, Any]:
    """Records a plan rejection and its human-provided rationale."""
    trace_id = state.get("trace_id", "-")
    feedback = state.get("human_feedback") or "no feedback provided"
    logger.warning("[%s] rejected: plan rejected  feedback=%r", trace_id, feedback)
    return {"error": f"Plan rejected by human. Feedback: {feedback}"}


# ── Executor ───────────────────────────────────────────────────────────────────

def executor_node(state: MMMGraphState) -> dict[str, Any]:
    """
    Executes all tool calls from the approved plan.

    Results accumulate in execution_log (via _append_list reducer), so if
    a retry path re-enters this node the full history is preserved.
    """
    trace_id = state.get("trace_id", "-")
    logger.info("[%s] executor: starting  plan_objective=%r", trace_id, (state.get("plan") or {}).get("objective", ""))

    try:
        collection = get_collection()
        tool_registry = build_tool_registry(collection)
        plan = TaskPlan(**state["plan"])
        execution_log = run_executor(plan, tool_registry)
        tools_used = [entry["tool_name"] for entry in execution_log]
        errors = [e["tool_name"] for e in execution_log if str(e.get("result", "")).startswith("ERROR")]

        if errors:
            logger.warning("[%s] executor: %d tool(s) failed  tools=%s", trace_id, len(errors), errors)
        logger.info("[%s] executor: done  tools=%d  errors=%d", trace_id, len(execution_log), len(errors))

        return {"execution_log": execution_log, "tools_used": tools_used}

    except Exception as exc:
        logger.error("[%s] executor: fatal error  error=%s", trace_id, exc, exc_info=True)
        return {"error": f"Executor failed: {exc}"}


# ── Synthesizer ────────────────────────────────────────────────────────────────

def synthesizer_node(state: MMMGraphState) -> dict[str, Any]:
    """
    Synthesizes tool results into a structured CopilotAnswer dict.

    Reads from state["execution_log"] which may contain results from
    multiple executor passes if retry logic ran.
    """
    trace_id = state.get("trace_id", "-")
    execution_log = state.get("execution_log") or []
    logger.info("[%s] synthesizer: starting  log_entries=%d", trace_id, len(execution_log))

    try:
        answer = run_synthesizer(state["question"], execution_log)
        logger.info(
            "[%s] synthesizer: done  insights=%d  chart=%s",
            trace_id,
            len(answer.get("insights", [])),
            answer.get("chart_type"),
        )
        return {"answer": answer}

    except Exception as exc:
        logger.error("[%s] synthesizer: failed  error=%s", trace_id, exc, exc_info=True)
        return {"error": f"Synthesizer failed: {exc}"}
