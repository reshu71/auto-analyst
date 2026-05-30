"""
src.graph — MMM LangGraph pipeline (FAANG-grade rewrite of mmm_graph.py).

Public API
----------
    from src.graph import run_with_approval, run_silent, GraphConfig, MMMGraphState

run_with_approval(question, config)
    Interactive pipeline with human plan-approval step (uses interrupt/resume).

run_silent(question, config)
    Programmatic pipeline; skips human approval entirely.

GraphConfig
    Dataclass that controls checkpointer path, retry limits, and approval flag.

MMMGraphState
    TypedDict that describes the full state flowing through the graph.
"""

from .runner import run_silent, run_with_approval
from .state import GraphConfig, MMMGraphState

__all__ = [
    "run_with_approval",
    "run_silent",
    "GraphConfig",
    "MMMGraphState",
]
