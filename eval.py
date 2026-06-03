"""
eval.py — MMM Copilot Evaluation Harness
=========================================
Scores the pipeline on 5 golden test cases across 3 dimensions:
  1. Tool Accuracy    — did the planner call the right tools?
  2. Keyword Coverage — did the answer mention the right concepts?
  3. LLM-as-Judge     — faithfulness, correctness, actionability (1-5 each)

Run:
    python eval.py
    python eval.py --save          # saves results to eval_results.json
    python eval.py --question-id gtc_001  # run single test case
"""

import re
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from litellm import completion
from dotenv import load_dotenv
from langfuse import get_client

from src.db import get_collection
from src.config import LLM_MODEL
from pipeline import run as run_pipeline

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")

langfuse = get_client()

# ─── Golden Dataset ───────────────────────────────────────────────
GOLDEN_DATASET = [
    {
        "id": "gtc_001",
        "category": "roi_comparison",
        "question": "What is the ROI for Doximity in Januvia's 2024 MMM, and how does it compare to the pharma industry benchmark?",
        "expected_tools": ["mmm_retriever", "benchmark_fetcher"],
        "expected_keywords": ["roi", "doximity", "januvia", "benchmark", "pharma"],
        "ground_truth": (
            "Doximity delivers strong ROI for Januvia in 2024, above the pharma industry "
            "midpoint benchmark of 2.7x but below the 75th percentile of 3.1x — "
            "a solid above-average performer."
        ),
    },
    {
        "id": "gtc_002",
        "category": "channel_ranking",
        "question": "Which HCP channel had the highest ROI for Keytruda in 2023?",
        "expected_tools": ["mmm_retriever"],
        "expected_keywords": ["roi", "hcp", "keytruda", "oncology"],
        "ground_truth": (
            "The highest ROI HCP channel for Keytruda in 2023 is identified with its "
            "spend and revenue contribution showing it as the most efficient HCP channel."
        ),
    },
    {
        "id": "gtc_003",
        "category": "benchmark",
        "question": "How does Medscape ROI compare across oncology brands, and is it above or below the oncology benchmark?",
        "expected_tools": ["mmm_retriever", "benchmark_fetcher"],
        "expected_keywords": ["roi", "medscape", "oncology", "benchmark"],
        "ground_truth": (
            "Medscape ROI varies across oncology brands with comparison to the oncology "
            "benchmark midpoint of 2.8x and 75th percentile of 3.2x."
        ),
    },
    {
        "id": "gtc_004",
        "category": "underperforming",
        "question": "Which channels are generating a negative return on investment (ROI below 1.0) for Dificid in 2024?",
        "expected_tools": ["mmm_retriever", "benchmark_fetcher"],
        "expected_keywords": ["roi", "dificid", "below", "underperforming"],
        "ground_truth": (
            "Channels with ROI below 1.0 for Dificid in 2024 are identified, "
            "with recommendations to reallocate budget to higher-performing channels."
        ),
    },
    {
        "id": "gtc_005",
        "category": "base_incremental",
        "question": "What is the base versus incremental revenue split for Lenvima in 2023, and what does it imply about media effectiveness?",
        "expected_tools": ["mmm_retriever"],
        "expected_keywords": ["incremental", "base", "lenvima", "media"],
        "ground_truth": (
            "Lenvima 2023 shows a near-even base/incremental split which is unusually "
            "media-driven for oncology. Most pharma brands sit at 60-70% base, so high "
            "incremental share suggests strong media contribution."
        ),
    },
]


# ─── Scorer 1 — Tool Accuracy ─────────────────────────────────────
def score_tool_accuracy(expected_tools: list, actual_tools: list) -> dict:
    """
    Precision / Recall / F1 on tool selection.
    - Precision: of tools called, how many were correct?
    - Recall:    of tools expected, how many were called?
    """
    expected_set = set(expected_tools)
    actual_set   = set(t for t in actual_tools if "chart" not in t)  # exclude chart tools from scoring

    correct  = expected_set & actual_set
    missed   = expected_set - actual_set
    extra    = actual_set   - expected_set

    precision = len(correct) / len(actual_set)   if actual_set   else 0.0
    recall    = len(correct) / len(expected_set) if expected_set else 0.0
    f1        = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "precision": round(precision, 2),
        "recall":    round(recall, 2),
        "f1":        round(f1, 2),
        "correct":   sorted(correct),
        "missed":    sorted(missed),
        "extra":     sorted(extra),
    }


# ─── Scorer 2 — Keyword Coverage ─────────────────────────────────
def score_keyword_coverage(expected_keywords: list, answer_text: str) -> dict:
    """
    What % of expected keywords appear in the answer?
    Score is scaled to 1-5 for consistent reporting.
    """
    answer_lower = answer_text.lower()
    found  = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missed = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    coverage = round(len(found) / len(expected_keywords), 2) if expected_keywords else 0.0

    return {
        "coverage": coverage,
        "score":    round(coverage * 5, 1),   # 0.0–1.0 → 0–5
        "found":    found,
        "missed":   missed,
    }


# ─── Scorer 3 — LLM-as-Judge ─────────────────────────────────────
JUDGE_PROMPT = """
You are an expert MMM (Marketing Mix Modeling) analyst evaluating an AI answer.

Question:
{question}

Ground Truth (what a correct answer looks like):
{ground_truth}

AI System Answer:
{answer}

Score the answer on these three dimensions. Return ONLY valid JSON — no markdown, no explanation.

{{
  "faithfulness":  <int 1-5>,
  "correctness":   <int 1-5>,
  "actionability": <int 1-5>,
  "reasoning":     "<one sentence explaining the scores>"
}}

Scoring guide:
- faithfulness  : 1 = hallucinated claims, 5 = all claims grounded in retrieved data
- correctness   : 1 = wrong conclusion, 5 = matches ground truth reasoning exactly
- actionability : 1 = vague or no recommendation, 5 = specific implementable action
"""

def score_with_llm_judge(question: str, ground_truth: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        answer=answer[:3000],   # truncate to avoid huge contexts
    )
    try:
        response = completion(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        scores = json.loads(raw)
        scores["composite"] = round(
            (scores["faithfulness"] + scores["correctness"] + scores["actionability"]) / 3,
            2,
        )
        return scores
    except Exception as e:
        logger.warning("LLM judge parse error: %s", e)
        return {
            "faithfulness": 0, "correctness": 0,
            "actionability": 0, "composite": 0.0,
            "reasoning": f"parse error: {e}",
        }


# ─── Extract Full Answer Text ─────────────────────────────────────
def extract_answer_text(answer: dict) -> str:
    """Combine all text fields from the pipeline answer into one string for scoring."""
    parts = [
        answer.get("summary", ""),
        answer.get("narrative", ""),
    ]
    for ins in answer.get("insights", []):
        parts.append(f"{ins.get('channel', '')} {ins.get('recommendation', '')}")
    return " ".join(p for p in parts if p).lower()


# ─── Single Test Case Eval ────────────────────────────────────────
def eval_single(test_case: dict, collection) -> dict:
    logger.info("Evaluating %s: %s", test_case["id"], test_case["question"][:60])

    # run pipeline
    try:
        answer = run_pipeline(test_case["question"], collection)
    except Exception as e:
        logger.error("Pipeline error on %s: %s", test_case["id"], e)
        return {
            "id":            test_case["id"],
            "category":      test_case["category"],
            "question":      test_case["question"],
            "error":         str(e),
            "tool_accuracy": {"f1": 0, "precision": 0, "recall": 0},
            "keyword_score": {"coverage": 0, "score": 0},
            "llm_judge":     {"faithfulness": 0, "correctness": 0, "actionability": 0, "composite": 0},
            "overall_score": 0.0,
        }

    answer_text  = extract_answer_text(answer)
    actual_tools = answer.get("tools_used", [])

    # score
    tool_score    = score_tool_accuracy(test_case["expected_tools"], actual_tools)
    keyword_score = score_keyword_coverage(test_case["expected_keywords"], answer_text)
    llm_score     = score_with_llm_judge(
        question=test_case["question"],
        ground_truth=test_case["ground_truth"],
        answer=answer_text,
    )

    # overall: weighted average (tool accuracy counts less — it's more mechanical)
    overall = round(
        (
            tool_score["f1"]        * 5  * 0.25 +   # tool accuracy  — 25% weight
            keyword_score["score"]       * 0.25 +   # keyword coverage — 25% weight
            llm_score["composite"]       * 0.50     # LLM judge       — 50% weight
        ),
        2,
    )

    result = {
        "id":             test_case["id"],
        "category":       test_case["category"],
        "question":       test_case["question"],
        "actual_tools":   actual_tools,
        "tool_accuracy":  tool_score,
        "keyword_score":  keyword_score,
        "llm_judge":      llm_score,
        "overall_score":  overall,
        "answer_summary": answer.get("summary", "")[:200],
    }

    logger.info(
        "%s — Tool F1: %.2f | Keyword: %.2f | Judge: %.2f | Overall: %.2f/5",
        test_case["id"],
        tool_score["f1"],
        keyword_score["coverage"],
        llm_score["composite"],
        overall,
    )

    # log eval score to Langfuse
    try:
        langfuse.create_score(
            name="overall_score",
            value=overall,
            comment=f"Tool F1: {tool_score['f1']} | Judge: {llm_score['composite']}",
        )
    except Exception as e:
        logger.debug("Langfuse score log skipped: %s", e)

    return result


# ─── Full Eval Run ────────────────────────────────────────────────
def run_eval(question_id: str = None, save: bool = False) -> list[dict]:
    collection = get_collection()

    dataset = GOLDEN_DATASET
    if question_id:
        dataset = [tc for tc in GOLDEN_DATASET if tc["id"] == question_id]
        if not dataset:
            logger.error("No test case found with id: %s", question_id)
            return []

    logger.info("Running eval on %d test cases", len(dataset))
    results = []

    for tc in dataset:
        result = eval_single(tc, collection)
        results.append(result)

    # flush langfuse after all runs
    langfuse.flush()

    print_report(results)

    if save:
        output_path = Path("eval_results.json")
        with open(output_path, "w") as f:
            json.dump({
                "run_at":  datetime.now().isoformat(),
                "model":   LLM_MODEL,
                "results": results,
            }, f, indent=2)
        logger.info("Results saved to %s", output_path)

    return results


# ─── Report Printer ───────────────────────────────────────────────
def print_report(results: list[dict]) -> None:
    if not results:
        print("No results to display.")
        return

    print(f"\n{'='*80}")
    print("MMM COPILOT — EVAL REPORT")
    print(f"{'='*80}")

    # per-test summary
    rows = []
    for r in results:
        if "error" in r:
            rows.append({
                "ID":          r["id"],
                "Category":    r["category"],
                "Tool F1":     "ERROR",
                "KW Cov":      "ERROR",
                "Faithful":    "ERROR",
                "Correct":     "ERROR",
                "Actionable":  "ERROR",
                "Overall /5":  0.0,
            })
        else:
            rows.append({
                "ID":          r["id"],
                "Category":    r["category"],
                "Tool F1":     r["tool_accuracy"]["f1"],
                "KW Cov":      r["keyword_score"]["coverage"],
                "Faithful":    r["llm_judge"]["faithfulness"],
                "Correct":     r["llm_judge"]["correctness"],
                "Actionable":  r["llm_judge"]["actionability"],
                "Overall /5":  r["overall_score"],
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    # aggregate stats
    numeric_results = [r for r in results if "error" not in r]
    if numeric_results:
        print(f"\n{'='*80}")
        print("AGGREGATE SCORES")
        print(f"{'='*80}")

        mean_tool    = sum(r["tool_accuracy"]["f1"]        for r in numeric_results) / len(numeric_results)
        mean_kw      = sum(r["keyword_score"]["coverage"]  for r in numeric_results) / len(numeric_results)
        mean_faith   = sum(r["llm_judge"]["faithfulness"]  for r in numeric_results) / len(numeric_results)
        mean_correct = sum(r["llm_judge"]["correctness"]   for r in numeric_results) / len(numeric_results)
        mean_action  = sum(r["llm_judge"]["actionability"] for r in numeric_results) / len(numeric_results)
        mean_overall = sum(r["overall_score"]              for r in numeric_results) / len(numeric_results)

        print(f"  Tool Accuracy F1    : {mean_tool:.2f}")
        print(f"  Keyword Coverage    : {mean_kw:.2f}")
        print(f"  Faithfulness        : {mean_faith:.2f} / 5")
        print(f"  Correctness         : {mean_correct:.2f} / 5")
        print(f"  Actionability       : {mean_action:.2f} / 5")
        print(f"\n  ► Mean Overall Score: {mean_overall:.2f} / 5")

        # per-category breakdown
        categories = list({r["category"] for r in numeric_results})
        if len(categories) > 1:
            print(f"\n{'='*80}")
            print("BY CATEGORY")
            print(f"{'='*80}")
            for cat in sorted(categories):
                cat_results = [r for r in numeric_results if r["category"] == cat]
                cat_mean    = sum(r["overall_score"] for r in cat_results) / len(cat_results)
                print(f"  {cat:<25} : {cat_mean:.2f} / 5  ({len(cat_results)} test case{'s' if len(cat_results) > 1 else ''})")

        # flag weak spots
        print(f"\n{'='*80}")
        print("WEAK SPOTS")
        print(f"{'='*80}")
        for r in numeric_results:
            issues = []
            if r["tool_accuracy"]["missed"]:
                issues.append(f"missed tools: {r['tool_accuracy']['missed']}")
            if r["keyword_score"]["coverage"] < 0.5:
                issues.append(f"low keyword coverage: {r['keyword_score']['missed']}")
            if r["llm_judge"]["faithfulness"] < 3:
                issues.append(f"low faithfulness: {r['llm_judge']['faithfulness']}/5")
            if r["llm_judge"]["correctness"] < 3:
                issues.append(f"low correctness: {r['llm_judge']['correctness']}/5")
            if issues:
                print(f"  {r['id']}: {' | '.join(issues)}")

        if not any(
            r["tool_accuracy"]["missed"] or
            r["keyword_score"]["coverage"] < 0.5 or
            r["llm_judge"]["faithfulness"] < 3 or
            r["llm_judge"]["correctness"] < 3
            for r in numeric_results
        ):
            print("  No major weak spots detected.")

    print(f"\n{'='*80}\n")


# ─── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MMM Copilot Eval Harness")
    parser.add_argument(
        "--question-id",
        type=str,
        default=None,
        help="Run a single test case by ID (e.g. gtc_001)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to eval_results.json"
    )
    args = parser.parse_args()

    run_eval(question_id=args.question_id, save=args.save)