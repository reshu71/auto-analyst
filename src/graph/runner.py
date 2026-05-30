"""
High-level entry points for the MMM LangGraph pipeline.

Public API
----------
run_with_approval(question, config) — interactive human-in-the-loop run
run_silent(question, config)        — programmatic / batch run (no interrupt)

Both functions:
  1. Build and compile the appropriate graph variant.
  2. Create a fresh thread_id for this run (enables parallel runs + replay).
  3. Invoke the graph and handle the interrupt/resume lifecycle.
  4. Return the final MMMGraphState dict.

All SQLite I/O is wrapped in context managers so the connection is always
closed cleanly, even on exception.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from pipeline import print_answer

from .graph import GraphConfig, build_graph_silent, build_graph_with_approval
from .state import MMMGraphState

logger = logging.getLogger("mmm.graph.runner")


# ── State factory ──────────────────────────────────────────────────────────────

def _initial_state(question: str) -> MMMGraphState:
    """Build a clean initial state dict for a new run."""
    return MMMGraphState(
        question=question,
        trace_id=str(uuid.uuid4()),
        plan=None,
        human_feedback=None,
        execution_log=[],
        tools_used=[],
        answer=None,
        error=None,
        retry_count=0,
    )


# ── Public entry points ────────────────────────────────────────────────────────

def run_with_approval(
    question: str,
    config: GraphConfig | None = None,
) -> dict[str, Any]:
    """
    Execute the full human-in-the-loop MMM pipeline.

    Flow
    ----
    Phase 1 — plan generation
        The graph runs planner → human_approval and then pauses at the interrupt.
        The interrupt payload is printed to stdout for the human reviewer.

    Phase 2 — resume
        The human types 'approve' or rejection feedback at the prompt.
        The graph resumes: executor → synthesizer → END (or rejected → END).

    Returns the final state dict.  Callers can inspect:
        result["answer"]  — structured CopilotAnswer (if approved + successful)
        result["error"]   — rejection message or node failure (if applicable)
    """
    cfg = config or GraphConfig()
    thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}
    initial_state = _initial_state(question)

    logger.info(
        "run_with_approval: starting  thread_id=%s  trace_id=%s",
        thread_id,
        initial_state["trace_id"],
    )

    with SqliteSaver.from_conn_string(cfg.checkpointer_path) as checkpointer:
        app = build_graph_with_approval(cfg).compile(checkpointer=checkpointer)

        # ── Phase 1: generate plan, pause at interrupt ─────────────────────
        result = app.invoke(initial_state, config=run_config)

        if "__interrupt__" not in result:
            # Planner errored before reaching interrupt, or approval was skipped
            logger.warning(
                "run_with_approval: graph completed without interrupt  error=%s",
                result.get("error"),
            )
            _log_and_print_outcome(result, question)
            return result

        payload = result["__interrupt__"][0].value
        _render_approval_prompt(payload)

        feedback = input("\nYour decision: ").strip().lower()
        logger.info("run_with_approval: received feedback=%r  thread_id=%s", feedback, thread_id)

        # ── Phase 2: resume with human decision ───────────────────────────
        final_result = app.invoke(Command(resume=feedback), config=run_config)

    logger.info("run_with_approval: complete  thread_id=%s", thread_id)
    _log_and_print_outcome(final_result, question)
    return final_result


def run_silent(
    question: str,
    config: GraphConfig | None = None,
) -> dict[str, Any]:
    """
    Execute the MMM pipeline without human approval.

    Use this for:
    - Batch processing pipelines
    - CI/test harnesses
    - Programmatic callers that implement their own approval workflow

    Returns the final state dict (same shape as run_with_approval).
    """
    base_config = config or GraphConfig()
    # Force approval off regardless of what the caller passed
    cfg = GraphConfig(
        checkpointer_path=base_config.checkpointer_path,
        max_retries=base_config.max_retries,
        use_human_approval=False,
        extra=base_config.extra,
    )
    thread_id = str(uuid.uuid4())
    run_config = {"configurable": {"thread_id": thread_id}}
    initial_state = _initial_state(question)

    logger.info(
        "run_silent: starting  thread_id=%s  trace_id=%s",
        thread_id,
        initial_state["trace_id"],
    )

    with SqliteSaver.from_conn_string(cfg.checkpointer_path) as checkpointer:
        app = build_graph_silent(cfg).compile(checkpointer=checkpointer)
        result = app.invoke(initial_state, config=run_config)

    logger.info("run_silent: complete  thread_id=%s", thread_id)
    _log_and_print_outcome(result, question)
    return result


# ── Private helpers ────────────────────────────────────────────────────────────

def _render_approval_prompt(payload: dict) -> None:
    """Print the interrupt payload in a human-readable format."""
    print("\n" + "=" * 60)
    print("MMM COPILOT — PLAN APPROVAL")
    print("=" * 60)
    print(f"Question  : {payload.get('question', '')}")
    print(f"Objective : {payload.get('objective', '')}")
    if payload.get("reasoning"):
        print(f"Reasoning : {payload['reasoning']}")
    print()
    print("Proposed steps:")
    for step in payload.get("steps", []):
        print(f"  {step}")
    print(f"\n{payload.get('message', '')}")


def _log_and_print_outcome(result: dict, question: str) -> None:
    """Log the final outcome and render it to stdout."""
    if result.get("answer"):
        logger.info("Pipeline succeeded  question=%r", question)
        print_answer(result["answer"])
    elif result.get("error"):
        logger.warning("Pipeline ended with error: %s", result["error"])
        print(f"\nPipeline ended: {result['error']}")
    else:
        logger.warning("Pipeline ended with no answer and no error  question=%r", question)


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    _question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "What is the ROI of HCP channels for oncology brands?"
    )
    run_with_approval(_question)
