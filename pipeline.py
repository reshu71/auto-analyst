import json
import logging
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from litellm import completion

# ─── Langfuse v4 (March 2026 release — OpenTelemetry-based) ──────
from langfuse import observe, get_client, propagate_attributes
from src.agentic_retriver import agentic_retriever as mmm_retriever
from src.config import LLM_MODEL
from src.db import get_collection
from src.query_parser import load_known_entities
from src.benchmark_fetcher import benchmark_fetcher, benchmark_all_channels
from src.scenario_simulator import scenario_simulator, simulate_portfolio
from src.chart_generator import (
    roi_bar_chart,
    spend_vs_revenue_chart,
    revenue_waterfall_chart,
    scenario_bar_chart,
    benchmark_comparison_chart,
    power_curve_chart,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Get the global Langfuse client — uses LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY from .env
langfuse = get_client()


# ─── Pydantic Models ──────────────────────────────────────────────
class SubTask(BaseModel):
    task:      str
    tool_name: str
    tool_args: dict

class TaskPlan(BaseModel):
    objective: str
    subtasks:  list[SubTask]
    reasoning: str

class ChannelInsight(BaseModel):
    channel:       str
    current_roi:   float
    benchmark_roi: Optional[float]
    recommendation: str
    confidence:    str

class CopilotAnswer(BaseModel):
    summary:    str
    insights:   list[ChannelInsight]
    narrative:  str
    chart_type: Optional[str]
    sources:    list[str]


# ─── Tool Registry ────────────────────────────────────────────────
def build_tool_registry(collection) -> dict:
    load_known_entities(collection)

    def _mmm_retriever(args: dict) -> str:
        return mmm_retriever(
            question=args["question"],
            collection=collection,
            n_results=args.get("n_results", 8),
        )

    def _benchmark_fetcher(args: dict) -> dict:
        if "channels" in args:
            return benchmark_all_channels(
                channels=args["channels"],
                sub_vertical=args.get("sub_vertical", "pharma"),
            )
        return benchmark_fetcher(
            channel=args["channel"],
            sub_vertical=args.get("sub_vertical", "pharma"),
            current_roi=args.get("current_roi"),
        )

    def _scenario_simulator(args: dict) -> dict:
        if "channels" in args and "budget_changes" in args:
            return simulate_portfolio(
                channels=args["channels"],
                budget_changes=args["budget_changes"],
            )
        return scenario_simulator(args)

    def _roi_bar_chart(args: dict) -> str:
        return roi_bar_chart(
            channels=args["channels"],
            brand=args.get("brand", ""),
            sub_vertical=args.get("sub_vertical", ""),
        )

    def _spend_vs_revenue_chart(args: dict) -> str:
        return spend_vs_revenue_chart(
            channels=args["channels"],
            brand=args.get("brand", ""),
        )

    def _revenue_waterfall_chart(args: dict) -> str:
        return revenue_waterfall_chart(
            model_summary=args["model_summary"],
            channels=args["channels"],
            brand=args.get("brand", ""),
        )

    def _scenario_bar_chart(args: dict) -> str:
        return scenario_bar_chart(scenario_results=args["scenario_results"])

    def _benchmark_comparison_chart(args: dict) -> str:
        return benchmark_comparison_chart(
            benchmark_results=args["benchmark_results"],
            sub_vertical=args.get("sub_vertical", ""),
        )

    def _power_curve_chart(args: dict) -> str:
        return power_curve_chart(
            channel=args["channel"],
            current_spend=args["current_spend"],
            current_revenue=args["current_revenue"],
        )

    return {
        "mmm_retriever":               _mmm_retriever,
        "benchmark_fetcher":           _benchmark_fetcher,
        "scenario_simulator":          _scenario_simulator,
        "roi_bar_chart":               _roi_bar_chart,
        "spend_vs_revenue_chart":      _spend_vs_revenue_chart,
        "revenue_waterfall_chart":     _revenue_waterfall_chart,
        "scenario_bar_chart":          _scenario_bar_chart,
        "benchmark_comparison_chart":  _benchmark_comparison_chart,
        "power_curve_chart":           _power_curve_chart,
    }


# ─── Planner ──────────────────────────────────────────────────────
PLANNER_SYSTEM = """
You are a planning agent for an MMM (Marketing Mix Modeling) copilot system.

Your job is to break a user's analytics question into a sequence of tool calls.

Available tools:
- mmm_retriever          → retrieves relevant MMM data from the vector store
                           args: { "question": str, "n_results": int }
- benchmark_fetcher      → returns industry benchmark ROI ranges for a channel
                           args single: { "channel": str, "sub_vertical": str }
                           NOTE: do NOT include "current_roi" — the synthesizer will
                           extract the actual ROI from mmm_retriever results and compare
                           it against the benchmark ranges returned here.
                           args batch:  { "channels": [{"name": str}], "sub_vertical": str }
- scenario_simulator     → simulates budget change impact on revenue
                           args single:    { "channel": str, "current_spend": float, "current_revenue_contribution": float, "budget_change_pct": float }
                           args portfolio: { "channels": [...], "budget_changes": {"channel_name": pct} }
- roi_bar_chart          → bar chart of ROI per channel
                           args: { "channels": [...], "brand": str, "sub_vertical": str }
- spend_vs_revenue_chart → scatter chart of spend vs revenue
                           args: { "channels": [...], "brand": str }
- revenue_waterfall_chart → waterfall of revenue decomposition
                           args: { "model_summary": {...}, "channels": [...], "brand": str }
- scenario_bar_chart     → before/after scenario chart
                           args: { "scenario_results": [...] }
- benchmark_comparison_chart → your ROI vs industry benchmarks
                           args: { "benchmark_results": [...], "sub_vertical": str }
- power_curve_chart      → diminishing returns curve for a channel
                           args: { "channel": str, "current_spend": float, "current_revenue": float }

Rules:
1. Always start with mmm_retriever to get relevant MMM data
2. Only include tools necessary for the question
3. If budget changes or scenarios are mentioned — include scenario_simulator
4. If benchmarks, "vs industry", underperforming channels, or ROI below a threshold are mentioned — always include benchmark_fetcher to provide industry context
5. Always end with ONE chart that best visualises the answer
6. For cross-brand or multi-channel questions (e.g. "across oncology brands", "all channels"), use n_results=15 or higher in mmm_retriever
7. sub_vertical must be one of: "oncology", "vaccines", "pharma". Infer it from the brand name if not stated — Keytruda/Lenvima/Lynparza/Qliftara/Welireg are oncology; Januvia/Dificid/Bridion/Belsomra are pharma
8. Return ONLY valid JSON — no markdown, no explanation

Return format:
{
  "objective": "one sentence describing what we are answering",
  "reasoning": "why you chose these tools in this order",
  "subtasks": [
    { "task": "plain English description", "tool_name": "tool_name", "tool_args": { ... } }
  ]
}
"""

# Key change in v4 — as_type="generation" auto-captures model, tokens, cost
@observe(name="planner", as_type="generation")
def run_planner(question: str) -> TaskPlan:
    logger.info("Planner: generating task plan")

    # In v4 — input and metadata go on get_client(), not langfuse_context
    langfuse.update_current_generation(
        input=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": question},
        ],
        model=LLM_MODEL,
    )

    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {"role": "user",   "content": question},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        parsed = json.loads(raw)
        plan   = TaskPlan(**parsed)

        logger.info("Planner: objective=%r | subtasks=%d", plan.objective, len(plan.subtasks))
        for i, s in enumerate(plan.subtasks, 1):
            logger.info("  step %d: %s → %s", i, s.task, s.tool_name)

        # v4 — pass usage_details for token tracking (replaces old "usage" field)
        langfuse.update_current_generation(
            output=raw,
            usage_details={
                "input":  response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total":  response.usage.total_tokens,
            },
            metadata={
                "objective":     plan.objective,
                "reasoning":     plan.reasoning,
                "num_subtasks":  len(plan.subtasks),
                "tools_planned": [s.tool_name for s in plan.subtasks],
            },
        )
        return plan

    except Exception as e:
        logger.warning("Planner parse error (%s) — falling back to retrieval only", e)
        fallback = TaskPlan(
            objective=question,
            reasoning="Fallback to retrieval only due to parse error",
            subtasks=[SubTask(
                task="Retrieve relevant MMM data",
                tool_name="mmm_retriever",
                tool_args={"question": question, "n_results": 8},
            )],
        )
        langfuse.update_current_generation(
            output=raw,
            level="WARNING",
            status_message=f"Parse error: {e}",
            metadata={"fallback": True},
        )
        return fallback


# ─── Single Tool Span ─────────────────────────────────────────────
@observe()
def _run_single_tool(subtask: SubTask, tool_registry: dict) -> dict:
    """
    Each tool runs as its own Langfuse span.
    The span name is set dynamically to the tool name.
    """
    langfuse.update_current_span(
        name=subtask.tool_name,
        input=subtask.tool_args,
    )

    tool_fn = tool_registry.get(subtask.tool_name)

    if not tool_fn:
        err = f"ERROR: tool {subtask.tool_name!r} not registered"
        logger.warning(err)
        langfuse.update_current_span(
            output={"error": err},
            level="ERROR",
            status_message=err,
        )
        return {"task": subtask.task, "tool_name": subtask.tool_name, "result": err}

    try:
        result = tool_fn(subtask.tool_args)

        # don't log base64 chart blobs
        is_chart   = "chart" in subtask.tool_name
        output_log = (
            "[chart base64 — not logged to Langfuse]"
            if is_chart
            else (str(result)[:500] + "…" if len(str(result)) > 500 else str(result))
        )
        langfuse.update_current_span(output=output_log)
        return {"task": subtask.task, "tool_name": subtask.tool_name, "result": result}

    except Exception as e:
        err = f"ERROR: {e}"
        logger.error("Tool %s raised: %s", subtask.tool_name, e)
        langfuse.update_current_span(
            output={"error": err},
            level="ERROR",
            status_message=err,
        )
        return {"task": subtask.task, "tool_name": subtask.tool_name, "result": err}


# ─── Executor ─────────────────────────────────────────────────────
@observe(name="executor")
def run_executor(plan: TaskPlan, tool_registry: dict) -> list[dict]:
    """
    Pure Python loop — no LLM calls.
    Each tool becomes a nested span via _run_single_tool.
    """
    langfuse.update_current_span(
        input={
            "objective": plan.objective,
            "subtasks":  [{"task": s.task, "tool_name": s.tool_name} for s in plan.subtasks],
        }
    )

    execution_log = []
    for i, subtask in enumerate(plan.subtasks, 1):
        logger.info("[%d/%d] %s → %s", i, len(plan.subtasks), subtask.task, subtask.tool_name)
        entry = _run_single_tool(subtask, tool_registry)
        execution_log.append(entry)

        preview = str(entry["result"])
        if len(preview) > 120:
            preview = preview[:120] + "…"
        logger.info("[%d/%d] done: %s", i, len(plan.subtasks), preview)

    errors = [
        e["tool_name"] for e in execution_log
        if isinstance(e["result"], str) and e["result"].startswith("ERROR")
    ]

    langfuse.update_current_span(
        output={
            "tools_executed": [e["tool_name"] for e in execution_log],
            "num_results":    len(execution_log),
            "errors":         errors,
        },
        level="WARNING" if errors else "DEFAULT",
    )
    return execution_log


# ─── Synthesizer ──────────────────────────────────────────────────
SYNTHESIZER_SYSTEM = """
You are an expert MMM (Marketing Mix Modeling) analyst synthesizing findings for a pharma commercial team.

You will receive:
- The original question
- Results from multiple analytical tools (retrieval, benchmarks, scenario simulations)

Your job is to write a clear, structured answer that a commercial director would find actionable.

Return ONLY valid JSON in this exact format — no markdown, no backticks:
{
  "summary": "2-3 sentence executive summary",
  "insights": [
    {
      "channel": "channel_name",
      "current_roi": 2.5,
      "benchmark_roi": 3.0,
      "recommendation": "one clear action for this channel",
      "confidence": "high | medium | low"
    }
  ],
  "narrative": "3-5 paragraph detailed analysis with specific numbers from the data",
  "chart_type": "roi_bar | waterfall | scenario | benchmark | scatter | power_curve | none",
  "sources": ["list of brands/models referenced in the answer"]
}

CRITICAL Rules — faithfulness:
- ONLY cite ROI values, spend figures, and revenue numbers that appear VERBATIM in the tool results below
- For benchmark comparisons: read the actual ROI from the mmm_retriever result, then compare it against benchmark_roi_mid and benchmark_p75 from the benchmark_fetcher result
- If the benchmark_fetcher result shows benchmark_roi_mid=2.7 and the retrieved brand ROI is 2.91, state "2.91x is above the industry midpoint of 2.7x" — do not guess or estimate
- If data is missing or ambiguous, say so explicitly rather than fabricating a number
- For cross-brand questions, list each brand and its specific retrieved value individually
- Keep recommendations specific and actionable
- If no channel data is available for insights, return an empty insights list
- narrative should read like a consultant's written analysis
"""

@observe(name="synthesizer", as_type="generation")
def run_synthesizer(question: str, execution_log: list[dict]) -> dict:
    logger.info("Synthesizer: building context from %d tool results", len(execution_log))

    context_parts = []
    chart_b64     = None

    for entry in execution_log:
        tool   = entry["tool_name"]
        result = entry["result"]

        if "chart" in tool and isinstance(result, str) and len(result) > 200:
            chart_b64 = result
            context_parts.append(f"[{tool}]: Chart generated successfully")
        else:
            result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            if len(result_str) > 3000:
                result_str = result_str[:3000] + "\n… (truncated)"
            context_parts.append(f"[{entry['task']}]\nTool: {tool}\nResult:\n{result_str}")

    context      = "\n\n---\n\n".join(context_parts)
    user_message = (
        f"Question: {question}\n\n"
        f"Tool Results:\n{context}\n\n"
        f"Synthesize a complete answer using the data above."
    )

    langfuse.update_current_generation(
        input=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
        model=LLM_MODEL,
    )

    response = completion(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIZER_SYSTEM},
            {"role": "user",   "content": user_message},
        ],
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```json|```", "", raw).strip()

    try:
        answer = json.loads(raw)
        logger.info("Synthesizer: %d insights | chart_type=%s",
                    len(answer.get("insights", [])), answer.get("chart_type"))
    except Exception as e:
        logger.warning("Synthesizer JSON parse error (%s) — raw text as narrative", e)
        answer = {
            "summary":    "Analysis complete.",
            "insights":   [],
            "narrative":  raw,
            "chart_type": "none",
            "sources":    [],
        }

    langfuse.update_current_generation(
        output=raw,
        usage_details={
            "input":  response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "total":  response.usage.total_tokens,
        },
        metadata={
            "num_insights": len(answer.get("insights", [])),
            "chart_type":   answer.get("chart_type"),
            "sources":      answer.get("sources", []),
            "has_chart":    chart_b64 is not None,
        },
    )
    answer["tools_used"] = [e["tool_name"] for e in execution_log]
    answer["chart_b64"] = chart_b64
    return answer


# ─── Full Pipeline ────────────────────────────────────────────────
@observe(name="mmm_pipeline")
def run(question: str, collection) -> dict:
    """
    Langfuse v4 trace structure:
      mmm_pipeline                ← outer trace (span)
        ├── planner               ← generation (tokens + cost auto-tracked)
        ├── executor              ← span
        │     ├── mmm_retriever   ← span
        │     ├── benchmark_fetcher ← span
        │     └── <chart tool>    ← span
        └── synthesizer           ← generation (tokens + cost auto-tracked)
    """
    langfuse.set_current_trace_io(input={"question": question})

    logger.info("=" * 60)
    logger.info("Pipeline start: %r", question)
    logger.info("=" * 60)

    with propagate_attributes(tags=["pipeline", "mmm"]):
        tool_registry = build_tool_registry(collection)
        plan          = run_planner(question)
        execution_log = run_executor(plan, tool_registry)
        answer        = run_synthesizer(question, execution_log)

    langfuse.set_current_trace_io(
        output={
            "summary":      answer.get("summary", ""),
            "chart_type":   answer.get("chart_type"),
            "num_insights": len(answer.get("insights", [])),
            "sources":      answer.get("sources", []),
        }
    )

    logger.info("Pipeline complete")
    return answer


# ─── Terminal Renderer ────────────────────────────────────────────
def print_answer(answer: dict) -> None:
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(answer.get("summary", ""))

    if answer.get("insights"):
        print(f"\n{'='*60}")
        print("CHANNEL INSIGHTS")
        print(f"{'='*60}")
        for ins in answer["insights"]:
            print(f"\n  {ins['channel'].upper()}")
            print(f"    Current ROI   : {ins.get('current_roi', 'N/A')}x")
            print(f"    Benchmark ROI : {ins.get('benchmark_roi', 'N/A')}x")
            print(f"    Confidence    : {ins.get('confidence', '')}")
            print(f"    Recommendation: {ins.get('recommendation', '')}")

    print(f"\n{'='*60}")
    print("NARRATIVE")
    print(f"{'='*60}")
    print(answer.get("narrative", ""))

    if answer.get("chart_b64"):
        print(f"\n{'='*60}")
        print(f"CHART: {answer.get('chart_type', 'generated')} (base64 ready for UI)")
        print(f"{'='*60}")

    if answer.get("sources"):
        print(f"\nSources: {', '.join(answer['sources'])}")


# ─── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    collection = get_collection()

    questions = [
        "What is the ROI of HCP channels for oncology brands and how do they compare to industry benchmarks?",
        "If I cut TV budget by 20% for vaccines and reinvest it in paid search, what happens to revenue?",
        "Which channels are underperforming for Keytruda and what should we do about them?",
    ]

    for q in questions:
        answer = run(q, collection)
        print_answer(answer)
        print("\n\n")

    # v4 — important for short-lived scripts — flushes events before exit
    langfuse.flush()