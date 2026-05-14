import json
import logging
import re
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel
from litellm import completion

from src.config import LLM_MODEL
from src.db import get_collection
from src.query_parser import mmm_retriever, load_known_entities
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
    channel:        str
    current_roi:    float
    benchmark_roi:  Optional[float]
    recommendation: str
    confidence:     str   # high / medium / low

class CopilotAnswer(BaseModel):
    summary:         str
    insights:        list[ChannelInsight]
    narrative:       str
    chart_type:      Optional[str]
    sources:         list[str]


# ─── Tool Registry ────────────────────────────────────────────────
def build_tool_registry(collection) -> dict:
    known_brands, known_channels = load_known_entities(collection)

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
            current_roi=args["current_roi"],
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
        "mmm_retriever":              _mmm_retriever,
        "benchmark_fetcher":          _benchmark_fetcher,
        "scenario_simulator":         _scenario_simulator,
        "roi_bar_chart":              _roi_bar_chart,
        "spend_vs_revenue_chart":     _spend_vs_revenue_chart,
        "revenue_waterfall_chart":    _revenue_waterfall_chart,
        "scenario_bar_chart":         _scenario_bar_chart,
        "benchmark_comparison_chart": _benchmark_comparison_chart,
        "power_curve_chart":          _power_curve_chart,
    }


# ─── Planner ──────────────────────────────────────────────────────
PLANNER_SYSTEM = """
You are a planning agent for an MMM (Marketing Mix Modeling) copilot system.

Your job is to break a user's analytics question into a sequence of tool calls.

Available tools:
- mmm_retriever         → retrieves relevant MMM data from the vector store
                          args: { "question": str, "n_results": int }
- benchmark_fetcher     → compares channel ROI vs industry benchmarks
                          args single: { "channel": str, "sub_vertical": str, "current_roi": float }
                          args batch:  { "channels": [{"name": str, "roi": float}], "sub_vertical": str }
- scenario_simulator    → simulates budget change impact on revenue
                          args single:    { "channel": str, "current_spend": float, "current_revenue_contribution": float, "budget_change_pct": float }
                          args portfolio: { "channels": [...], "budget_changes": {"channel": pct} }
- roi_bar_chart         → bar chart of ROI per channel
                          args: { "channels": [...], "brand": str, "sub_vertical": str }
- spend_vs_revenue_chart → scatter chart of spend vs revenue
                          args: { "channels": [...], "brand": str }
- revenue_waterfall_chart → waterfall chart of revenue decomposition
                          args: { "model_summary": {...}, "channels": [...], "brand": str }
- scenario_bar_chart    → before/after scenario chart
                          args: { "scenario_results": [...] }
- benchmark_comparison_chart → your ROI vs industry benchmarks chart
                          args: { "benchmark_results": [...], "sub_vertical": str }
- power_curve_chart     → diminishing returns curve for a channel
                          args: { "channel": str, "current_spend": float, "current_revenue": float }

Rules:
1. Always start with mmm_retriever to get the relevant MMM data
2. Only include tools that are necessary for the question
3. If the question mentions budget changes or scenarios — include scenario_simulator
4. If the question asks for benchmarks or "how are we doing vs industry" — include benchmark_fetcher
5. Always end with ONE chart that best visualises the answer
6. Return ONLY valid JSON — no markdown, no explanation

Return format:
{
  "objective": "one sentence describing what we are answering",
  "reasoning": "why you chose these tools in this order",
  "subtasks": [
    { "task": "plain English description", "tool_name": "tool_name", "tool_args": { ... } }
  ]
}
"""

def run_planner(question: str) -> TaskPlan:
    logger.info("Planner: generating task plan for question")
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
        plan = TaskPlan(**parsed)
        logger.info(
            "Planner: objective=%r | subtasks=%d | reasoning=%r",
            plan.objective, len(plan.subtasks), plan.reasoning,
        )
        for i, s in enumerate(plan.subtasks, 1):
            logger.info("  step %d: %s → %s", i, s.task, s.tool_name)
        return plan
    except Exception as e:
        logger.warning("Planner parse error (%s) — falling back to retrieval only", e)
        return TaskPlan(
            objective=question,
            reasoning="Fallback to retrieval only",
            subtasks=[SubTask(
                task="Retrieve relevant MMM data",
                tool_name="mmm_retriever",
                tool_args={"question": question, "n_results": 8},
            )],
        )


# ─── Executor ─────────────────────────────────────────────────────
def run_executor(plan: TaskPlan, tool_registry: dict) -> list[dict]:
    """Pure Python loop — no LLM calls. Runs each subtask, collects results."""
    execution_log = []
    n = len(plan.subtasks)

    for i, subtask in enumerate(plan.subtasks, 1):
        logger.info("[%d/%d] executing %r via %s", i, n, subtask.task, subtask.tool_name)

        tool_fn = tool_registry.get(subtask.tool_name)
        if not tool_fn:
            logger.warning("Tool not found: %s", subtask.tool_name)
            execution_log.append({
                "task":      subtask.task,
                "tool_name": subtask.tool_name,
                "result":    f"ERROR: tool {subtask.tool_name!r} not registered",
            })
            continue

        try:
            result = tool_fn(subtask.tool_args)
            # log a brief preview — avoid dumping huge base64 blobs
            preview = str(result)[:120] + "…" if len(str(result)) > 120 else str(result)
            logger.info("[%d/%d] done — result preview: %s", i, n, preview)
            execution_log.append({
                "task":      subtask.task,
                "tool_name": subtask.tool_name,
                "result":    result,
            })
        except Exception as e:
            logger.error("[%d/%d] tool %s raised: %s", i, n, subtask.tool_name, e)
            execution_log.append({
                "task":      subtask.task,
                "tool_name": subtask.tool_name,
                "result":    f"ERROR: {e}",
            })

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
  "chart_type": "which chart was generated (roi_bar | waterfall | scenario | benchmark | scatter | power_curve | none)",
  "sources": ["list of brands/models referenced in the answer"]
}

Rules:
- Use actual numbers from the tool results — no made-up figures
- Keep recommendations specific and actionable
- If no channel data is available for insights, return an empty insights list
- narrative should read like a consultant's written analysis
"""

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
            logger.debug("Synthesizer: extracted chart from %s (%d bytes)", tool, len(result))
        else:
            result_str = json.dumps(result, indent=2) if isinstance(result, dict) else str(result)
            if len(result_str) > 3000:
                result_str = result_str[:3000] + "\n… (truncated)"
            context_parts.append(
                f"[{entry['task']}]\nTool: {tool}\nResult:\n{result_str}"
            )

    context = "\n\n---\n\n".join(context_parts)
    user_message = f"Question: {question}\n\nTool Results:\n{context}\n\nSynthesize a complete answer using the data above."

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
        logger.info(
            "Synthesizer: answer generated — %d insights, chart_type=%s",
            len(answer.get("insights", [])),
            answer.get("chart_type"),
        )
    except Exception as e:
        logger.warning("Synthesizer JSON parse error (%s) — using raw text as narrative", e)
        answer = {
            "summary":    "Analysis complete.",
            "insights":   [],
            "narrative":  raw,
            "chart_type": "none",
            "sources":    [],
        }

    answer["chart_b64"] = chart_b64
    return answer


# ─── Full Pipeline ────────────────────────────────────────────────
def run(question: str, collection) -> dict:
    """
    Full MMM Copilot pipeline:
      1. run_planner()    — LLM call #1 → TaskPlan
      2. run_executor()   — pure Python, no LLM, runs tools
      3. run_synthesizer() — LLM call #2 → CopilotAnswer + chart_b64
    """
    logger.info("=" * 60)
    logger.info("Pipeline start: %r", question)
    logger.info("=" * 60)

    tool_registry = build_tool_registry(collection)

    plan          = run_planner(question)
    execution_log = run_executor(plan, tool_registry)
    answer        = run_synthesizer(question, execution_log)

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
            print(f"    Current ROI  : {ins.get('current_roi', 'N/A')}x")
            print(f"    Benchmark ROI: {ins.get('benchmark_roi', 'N/A')}x")
            print(f"    Confidence   : {ins.get('confidence', '')}")
            print(f"    Recommendation: {ins.get('recommendation', '')}")

    print(f"\n{'='*60}")
    print("NARRATIVE")
    print(f"{'='*60}")
    print(answer.get("narrative", ""))

    if answer.get("chart_b64"):
        print(f"\n{'='*60}")
        print(f"CHART: {answer.get('chart_type', 'generated')} (base64 available)")
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
