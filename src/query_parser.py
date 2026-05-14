import re
import json
import logging
from litellm import completion
from .config import CURRENT_YEAR, LAST_YEAR, LLM_MODEL

logger = logging.getLogger(__name__)


def load_known_entities(collection) -> tuple[set, set]:
    all_metadata = collection.get()["metadatas"]
    brands = {meta["brand"] for meta in all_metadata if meta.get("brand")}
    channels = {
        meta["channel"] for meta in all_metadata
        if meta.get("channel") and meta["channel"] != "all"
    }
    return brands, channels


def regex_parser(question: str, known_brands: set, known_channels: set) -> dict:
    q = question.lower()

    found_brands = {b for b in known_brands if re.search(rf"\b{re.escape(b)}\b", q)}

    year_match = re.findall(r"\b(202[0-9])\b", question)
    if year_match:
        year = year_match[0]
    elif "last year" in q:
        year = LAST_YEAR
    elif "this year" in q:
        year = CURRENT_YEAR
    else:
        year = None

    found_channels = {c for c in known_channels if re.search(rf"\b{re.escape(c)}\b", q)}

    if "hcp" in q:
        category = "hcp"
    elif "consumer" in q or "hcc" in q:
        category = "consumer"
    else:
        category = None

    if "oncology" in q:
        sub_vertical = "oncology"
    elif "vaccine" in q or "vaccines" in q:
        sub_vertical = "vaccines"
    elif "pharma" in q:
        sub_vertical = "pharma"
    else:
        sub_vertical = None

    return {
        "brands": found_brands,
        "year": year,
        "channel": found_channels,
        "category": category,
        "sub_vertical": sub_vertical,
    }


def llm_parser(question: str, known_brands: set, known_channels: set) -> dict:
    prompt = f"""
Extract structured filters from this MMM analytics question.
Return ONLY a valid JSON object with these exact keys — no explanation, no markdown:

{{
  "brands":       [],
  "year":         null,
  "channel":      [],
  "category":     null,
  "sub_vertical": null
}}

Rules:
- brands: list of brand names from this list only: {sorted(known_brands)}
- year: 4-digit string like "2024", or null. Today is {CURRENT_YEAR} so "last year" = {LAST_YEAR}
- channel: list of channel/vendor names from this list only: {sorted(known_channels)}
- category: "hcp" or "consumer" or null
- sub_vertical: "oncology" or "vaccines" or "pharma" or null

Question: {question}
"""
    response = completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = re.sub(r"```json|```", "", response.choices[0].message.content.strip()).strip()

    try:
        parsed = json.loads(raw)
        return {
            "brands":       set(parsed.get("brands", [])),
            "year":         parsed.get("year"),
            "channel":      set(parsed.get("channel", [])),
            "category":     parsed.get("category"),
            "sub_vertical": parsed.get("sub_vertical"),
        }
    except json.JSONDecodeError:
        logger.warning("LLM parser returned invalid JSON — using empty filters")
        return {"brands": set(), "year": None, "channel": set(), "category": None, "sub_vertical": None}


def parse_query(question: str, known_brands: set, known_channels: set) -> dict:
    filters = regex_parser(question, known_brands, known_channels)
    if not any(filters.values()):
        logger.debug("Regex found nothing — falling back to LLM parser")
        filters = llm_parser(question, known_brands, known_channels)
    return filters


def build_where_clause(filters: dict) -> dict:
    conditions = []

    if filters.get("brands"):
        brands_list = list(filters["brands"])
        conditions.append(
            {"brand": {"$eq": brands_list[0]}} if len(brands_list) == 1
            else {"brand": {"$in": brands_list}}
        )

    if filters.get("year"):
        conditions.append({"year": {"$eq": filters["year"]}})

    if filters.get("channel"):
        channels_list = list(filters["channel"])
        conditions.append(
            {"channel": {"$eq": channels_list[0]}} if len(channels_list) == 1
            else {"channel": {"$in": channels_list}}
        )

    if filters.get("category"):
        conditions.append({"category": {"$eq": filters["category"]}})

    if filters.get("sub_vertical"):
        conditions.append({"sub_vertical": {"$eq": filters["sub_vertical"]}})

    if not conditions:
        return {}
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def mmm_retriever(question: str, collection, n_results: int = 10) -> str:
    known_brands, known_channels = load_known_entities(collection)
    filters = parse_query(question, known_brands, known_channels)
    logger.debug("Filters: %s", filters)

    where = build_where_clause(filters)
    logger.debug("Where clause: %s", where)

    results = collection.query(
        query_texts=[question],
        n_results=n_results,
        **({"where": where} if where else {}),
    )

    if not results["documents"][0]:
        return "No relevant MMM data found for this query."

    parts = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        parts.append(
            f"--- Result {i} ---\n"
            f"Brand: {meta['brand']} | Sub-vertical: {meta['sub_vertical']} | "
            f"Year: {meta['year']} | Type: {meta['type']}\n"
            f"Channel: {meta['channel']} | Category: {meta['category']}\n"
            f"Content: {doc}"
        )

    return "\n\n".join(parts)
