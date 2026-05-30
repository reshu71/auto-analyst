"""
Graph assembly for the MMM LangGraph pipeline.

Two factory functions are exposed:

  build_graph_with_approval()
      planner → human_approval --approve--> executor → synthesizer → END
                                --reject --> rejected              → END
      Uses interrupt() so the graph pauses and a human can review/approve
      the plan before execution begins.  Requires a persistent checkpointer
      (SqliteSaver) to survive the suspension.

  build_graph_silent()
      planner → executor → synthesizer → END
      No interrupt — safe for programmatic / batch callers.

Both factories return a compiled LangGraph CompiledGraph whose checkpointer
is injected by the caller (runner.py) so infrastructure concerns stay out of
the graph definition.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .nodes import (
    executor_node,
    human_approval_node,
    planner_node,
    rejected_node,
    synthesizer_node,
)
from .routing import (
    EXECUTOR,
    REJECTED,
    SYNTHESIZER,
    route_after_approval,
    route_after_executor,
    route_after_planner,
    route_after_planner_no_approval,
)
from .state import GraphConfig, MMMGraphState


def build_graph_with_approval(config: GraphConfig | None = None) -> StateGraph:
    """
    Returns an uncompiled StateGraph that includes the human-approval interrupt.
    Compile it with a checkpointer before invoking:

        saver = SqliteSaver.from_conn_string(config.checkpointer_path)
        app = build_graph_with_approval(config).compile(checkpointer=saver)
    """
    _ = config or GraphConfig()
    builder = StateGraph(MMMGraphState)

    # ── Register nodes ────────────────────────────────────────────────────────
    builder.add_node("planner", planner_node)
    builder.add_node("human_approval", human_approval_node)
    builder.add_node("rejected", rejected_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    builder.set_entry_point("planner")

    # ── Edges ─────────────────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {"human_approval": "human_approval", "__end__": END},
    )
    builder.add_conditional_edges(
        "human_approval",
        route_after_approval,
        {EXECUTOR: "executor", REJECTED: "rejected"},
    )
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {SYNTHESIZER: "synthesizer", "__end__": END},
    )
    builder.add_edge("synthesizer", END)
    builder.add_edge("rejected", END)

    return builder


def build_graph_silent(config: GraphConfig | None = None) -> StateGraph:
    """
    Returns an uncompiled StateGraph without human approval.
    The planner routes directly to the executor on success.
    """
    _ = config or GraphConfig()
    builder = StateGraph(MMMGraphState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("planner")

    builder.add_conditional_edges(
        "planner",
        route_after_planner_no_approval,
        {EXECUTOR: "executor", "__end__": END},
    )
    builder.add_conditional_edges(
        "executor",
        route_after_executor,
        {SYNTHESIZER: "synthesizer", "__end__": END},
    )
    builder.add_edge("synthesizer", END)

    return builder


def compile_in_memory(*, with_approval: bool = True, config: GraphConfig | None = None):
    """
    Convenience helper for unit tests and notebooks.
    Uses MemorySaver — state is lost when the process exits.
    """
    builder = (
        build_graph_with_approval(config)
        if with_approval
        else build_graph_silent(config)
    )
    return builder.compile(checkpointer=MemorySaver())
