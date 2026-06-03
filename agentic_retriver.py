"""
agentic_retriever.py — Iterative RAG retriever for MMM Copilot
===============================================================
Replaces the single-shot mmm_retriever() with an iterative loop that:
  1. Rewrites the query for better semantic coverage
  2. Checks keyword relevance
  3. Uses LLM-as-judge for deep relevance assessment
  4. Refines the query based on identified gaps
  5. Suggests corrections for unknown entities (did you mean?)

Max retries: 3 (configurable)
"""

import re
import json
from difflib import get_close_matches
from dotenv import load_dotenv
from litellm import completion

from src.config import LLM_MODEL
from src.query_parser import parse_query, build_where_clause, load_known_entities

load_dotenv()


# ─── Step 1 — Query Rewriter ──────────────────────────────────────
REWRITE_SYSTEM_PROMPT = """You are a search query optimizer for a pharma MMM vector database.
Rewrite the question as a dense keyword string (15-25 words)
optimized for semantic similarity search.
Include: brand names, channel names, metrics, time periods, therapeutic area.
Return ONLY the rewritten query string — no explanation."""


def rewrite_query(question: str, attempt_number: int, gap_description: str = "") -> str:
    """
    Rewrites the question into a dense keyword string for ChromaDB semantic search.
    attempt 1 — semantic expansion only
    attempt 2+ — gap-aware rewrite targeting what was missing
    """
    if attempt_number == 1:
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user",   "content": f"Question: {question}"},
        ]
    else:
        retry_content = (
            f"The original question is: {question}\n\n"
            f"The previous search was insufficient. The gap identified was: {gap_description}\n\n"
            f"Rewrite the original question to specifically target this missing information."
        )
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user",   "content": retry_content},
        ]

    response = completion(model=LLM_MODEL, messages=messages)
    return response.choices[0].message.content.strip()


# ─── Step 2 — Keyword Check ───────────────────────────────────────
def keyword_check(results: str, filters: dict) -> tuple[bool, str]:
    """
    Fast, cheap check — do the retrieved results mention the right entities?
    Returns (is_sufficient, gap_description)
    """
    passed  = []
    missing = []
    passed_checks = 0
    total_checks  = 0

    # brands — set of strings
    if filters.get("brands"):
        total_checks += 1
        if any(b.lower() in results.lower() for b in filters["brands"]):
            passed.append("brand")
            passed_checks += 1
        else:
            missing.append(f"brand: {', '.join(filters['brands'])}")

    # year — single string
    if filters.get("year"):
        total_checks += 1
        if filters["year"] in results:
            passed.append("year")
            passed_checks += 1
        else:
            missing.append(f"year: {filters['year']}")

    # channel — set of strings
    if filters.get("channel"):
        total_checks += 1
        if any(c.lower() in results.lower() for c in filters["channel"]):
            passed.append("channel")
            passed_checks += 1
        else:
            missing.append(f"channel: {', '.join(filters['channel'])}")

    # sub_vertical — single string
    if filters.get("sub_vertical"):
        total_checks += 1
        if filters["sub_vertical"].lower() in results.lower():
            passed.append("sub_vertical")
            passed_checks += 1
        else:
            missing.append(f"sub_vertical: {filters['sub_vertical']}")

    if total_checks == 0:
        return True, "no specific filters to check"

    if passed_checks / total_checks >= 0.5:
        return True, "results look relevant"
    else:
        return False, f"missing: {', '.join(missing)}"


# ─── Step 3 — LLM Relevance Judge ────────────────────────────────
JUDGE_SYSTEM_PROMPT = """You are a data sufficiency auditor for a pharma MMM analytics system.
Given a question and retrieved data chunks, decide if the chunks contain
enough specific information to answer the question accurately.
You must be strict — if key numbers, brand names, or channel metrics are
missing or belong to the wrong brand/year, return insufficient.
Return ONLY valid JSON:
{
  "sufficient": true or false,
  "gap": "what is missing or empty string if sufficient"
}"""


def llm_relevance_judge(question: str, results: str) -> tuple[bool, str]:
    """
    LLM-as-judge — deep check whether retrieved chunks are sufficient.
    Returns (is_sufficient, gap_description).
    Defaults to True on parse errors to avoid unnecessary retries.
    """
    user_message = (
        f"Question: {question}\n\n"
        f"Retrieved chunks:\n{results[:2000]}"
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_message},
    ]

    response = completion(model=LLM_MODEL, messages=messages)
    raw      = response.choices[0].message.content.strip()
    raw      = re.sub(r"```json|```", "", raw).strip()

    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return result.get("sufficient", True), result.get("gap", "")
        except json.JSONDecodeError:
            return True, ""   # judge failed — assume sufficient
    return True, ""           # no JSON found — assume sufficient


# ─── Step 4 — Did You Mean ────────────────────────────────────────
def did_you_mean(failed_entity: str, known_entities: list) -> str:
    """
    Suggests the closest known entity using fuzzy matching.
    """
    matches = get_close_matches(failed_entity, known_entities, n=1, cutoff=0.6)
    if matches:
        return f"Did you mean '{matches[0]}'?"
    return f"No similar entity found for '{failed_entity}'."


# ─── Step 5 — Format Results ──────────────────────────────────────
def format_results(results: dict) -> str:
    """
    Converts ChromaDB query results into a readable string for the LLM.
    Same format as the original mmm_retriever().
    """
    parts = []
    for i, (doc, meta) in enumerate(
        zip(results["documents"][0], results["metadatas"][0]), 1
    ):
        parts.append(
            f"--- Result {i} ---\n"
            f"Brand: {meta['brand']} | Sub-vertical: {meta['sub_vertical']} | "
            f"Year: {meta['year']} | Type: {meta['type']}\n"
            f"Channel: {meta['channel']} | Category: {meta['category']}\n"
            f"Content: {doc}"
        )
    return "\n\n".join(parts)


# ─── Step 6 — Agentic Retriever (main) ───────────────────────────
def agentic_retriever(question: str, collection, max_retries: int = 3) -> str:
    """
    Iterative RAG retriever.

    Loop (up to max_retries):
      1. Rewrite query (gap-aware on retry)
      2. Query ChromaDB with filters
      3. If zero results → did_you_mean and return
      4. Keyword check → if fails, retry with gap
      5. LLM judge → if sufficient, return results
      6. If not sufficient, update gap and retry

    Returns best results found, or a helpful error message.
    """
    known_brands, known_channels = load_known_entities(collection)
    filters         = parse_query(question, known_brands, known_channels)
    gap_description = ""
    best_results    = ""
    attempt         = 1

    while attempt <= max_retries:
        print(f"\n[Agentic Retriever] Attempt {attempt}/{max_retries}")

        # ── 1. Rewrite query ─────────────────────────────────────
        rewritten_query = rewrite_query(question, attempt, gap_description)
        print(f"  Rewritten query: {rewritten_query}")

        # ── 2. Build where clause and query ChromaDB ─────────────
        where = build_where_clause(filters, question)

        if where:
            results = collection.query(
                query_texts=[rewritten_query],
                n_results=10,
                where=where,
            )
        else:
            results = collection.query(
                query_texts=[rewritten_query],
                n_results=10,
            )

        # ── 3. Zero results → did you mean? ──────────────────────
        if not results["documents"][0]:
            print("  Zero results returned.")
            if filters.get("brands"):
                failed_entity = list(filters["brands"])[0]
                suggestion    = did_you_mean(failed_entity, list(known_brands))
                return f"No data found for '{failed_entity}'. {suggestion}"
            return "No relevant MMM data found for this query."

        # ── 4. Format results ─────────────────────────────────────
        formatted_results = format_results(results)
        best_results      = formatted_results   # track best so far

        # ── 5. Keyword check ──────────────────────────────────────
        kw_sufficient, gap_description = keyword_check(formatted_results, filters)
        print(f"  Keyword check: {'✓' if kw_sufficient else '✗'} — {gap_description}")

        if not kw_sufficient:
            attempt += 1
            continue

        # ── 6. LLM judge ──────────────────────────────────────────
        llm_sufficient, gap_description = llm_relevance_judge(question, formatted_results)
        print(f"  LLM judge: {'✓ sufficient' if llm_sufficient else '✗ insufficient'}")

        if llm_sufficient:
            print(f"  Retrieval successful on attempt {attempt}.")
            return formatted_results

        print(f"  Gap identified: {gap_description}")
        attempt += 1

    # ── Max retries hit — return best seen ────────────────────────
    print(f"  Max retries ({max_retries}) reached. Returning best results found.")
    return (
        best_results
        if best_results
        else f"After {max_retries} attempts, insufficient data found. Last gap: {gap_description}"
    )


# ─── Entry Point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import chromadb
    from chromadb.utils import embedding_functions
    from src.config import COLLECTION_NAME, MODEL_NAME

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=MODEL_NAME
    )
    client     = chromadb.PersistentClient(path="./mmm_vectorstore")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    test_questions = [
        # "What is the ROI for Doximity in Januvia's 2024 MMM?",
        # "Which HCP channel had the highest ROI for Keytruda in 2023?",
        # "What is the base versus incremental revenue split for Lenvima in 2023?",
        "What is the ROI for januvea in 2024?",   # typo — tests did_you_mean
    ]

    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print(f"{'='*60}")
        result = agentic_retriever(q, collection)
        print(f"\nResult preview:\n{result[:500]}...")