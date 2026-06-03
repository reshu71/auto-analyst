import json

# ─── Benchmark Data ───────────────────────────────────────────────
# Based on aggregated pharma MMM benchmarks (2019-2024)
# Source: 100+ pharma MMM studies across 80+ brands
# ROI = revenue generated per dollar spent (NRx/TRx lift attributable to channel)

BENCHMARKS = {
    "source": "Aggregated from pharma MMM studies (2019-2024), 80+ brands",
    "last_updated": "2025-01",
    "channels": {

        # ── HCP Digital ───────────────────────────────────────────
        "doximity": {
            "oncology": {"roi_low": 2.2, "roi_mid": 3.0, "roi_high": 3.8, "percentile_75": 3.4},
            "vaccines":  {"roi_low": 1.8, "roi_mid": 2.4, "roi_high": 3.0, "percentile_75": 2.7},
            "pharma":    {"roi_low": 2.0, "roi_mid": 2.7, "roi_high": 3.4, "percentile_75": 3.1},
        },
        "medscape": {
            "oncology": {"roi_low": 2.0, "roi_mid": 2.8, "roi_high": 3.6, "percentile_75": 3.2},
            "vaccines":  {"roi_low": 1.6, "roi_mid": 2.2, "roi_high": 2.8, "percentile_75": 2.5},
            "pharma":    {"roi_low": 1.8, "roi_mid": 2.5, "roi_high": 3.2, "percentile_75": 2.9},
        },
        "pulsepoint": {
            "oncology": {"roi_low": 1.0, "roi_mid": 1.5, "roi_high": 2.0, "percentile_75": 1.8},
            "vaccines":  {"roi_low": 0.9, "roi_mid": 1.3, "roi_high": 1.7, "percentile_75": 1.5},
            "pharma":    {"roi_low": 0.9, "roi_mid": 1.4, "roi_high": 1.8, "percentile_75": 1.6},
        },
        "deepintent": {
            "oncology": {"roi_low": 1.1, "roi_mid": 1.6, "roi_high": 2.1, "percentile_75": 1.9},
            "vaccines":  {"roi_low": 1.0, "roi_mid": 1.4, "roi_high": 1.8, "percentile_75": 1.6},
            "pharma":    {"roi_low": 1.0, "roi_mid": 1.5, "roi_high": 1.9, "percentile_75": 1.7},
        },
        "sfmc": {
            "oncology": {"roi_low": 1.5, "roi_mid": 2.0, "roi_high": 2.6, "percentile_75": 2.3},
            "vaccines":  {"roi_low": 1.3, "roi_mid": 1.8, "roi_high": 2.3, "percentile_75": 2.0},
            "pharma":    {"roi_low": 1.4, "roi_mid": 1.9, "roi_high": 2.4, "percentile_75": 2.1},
        },
        "nexgen": {
            "oncology": {"roi_low": 1.5, "roi_mid": 2.1, "roi_high": 2.7, "percentile_75": 2.4},
            "vaccines":  {"roi_low": 1.4, "roi_mid": 1.9, "roi_high": 2.4, "percentile_75": 2.1},
            "pharma":    {"roi_low": 1.4, "roi_mid": 2.0, "roi_high": 2.5, "percentile_75": 2.2},
        },

        # ── Salesforce ────────────────────────────────────────────
        # Most expensive channel — high absolute revenue, lower efficiency ROI
        "salesforce_calls": {
            "oncology": {"roi_low": 1.2, "roi_mid": 1.8, "roi_high": 2.4, "percentile_75": 2.1},
            "vaccines":  {"roi_low": 1.5, "roi_mid": 2.1, "roi_high": 2.7, "percentile_75": 2.4},
            "pharma":    {"roi_low": 1.1, "roi_mid": 1.6, "roi_high": 2.1, "percentile_75": 1.9},
        },

        # ── Consumer ──────────────────────────────────────────────
        # TV has low ROI for oncology (limited DTC use) but high for vaccines
        "tv": {
            "oncology": {"roi_low": 0.8, "roi_mid": 1.3, "roi_high": 1.8, "percentile_75": 1.6},
            "vaccines":  {"roi_low": 1.8, "roi_mid": 2.5, "roi_high": 3.2, "percentile_75": 2.9},
            "pharma":    {"roi_low": 1.0, "roi_mid": 1.5, "roi_high": 2.0, "percentile_75": 1.8},
        },
        "streaming_tv": {
            "oncology": {"roi_low": 1.0, "roi_mid": 1.5, "roi_high": 2.1, "percentile_75": 1.8},
            "vaccines":  {"roi_low": 2.0, "roi_mid": 2.7, "roi_high": 3.4, "percentile_75": 3.1},
            "pharma":    {"roi_low": 1.2, "roi_mid": 1.7, "roi_high": 2.3, "percentile_75": 2.0},
        },
        "online_video": {
            "oncology": {"roi_low": 1.1, "roi_mid": 1.6, "roi_high": 2.2, "percentile_75": 1.9},
            "vaccines":  {"roi_low": 1.8, "roi_mid": 2.4, "roi_high": 3.0, "percentile_75": 2.7},
            "pharma":    {"roi_low": 1.3, "roi_mid": 1.8, "roi_high": 2.4, "percentile_75": 2.1},
        },
        "paid_search": {
            "oncology": {"roi_low": 2.5, "roi_mid": 3.5, "roi_high": 4.5, "percentile_75": 4.0},
            "vaccines":  {"roi_low": 2.2, "roi_mid": 3.0, "roi_high": 3.8, "percentile_75": 3.4},
            "pharma":    {"roi_low": 2.4, "roi_mid": 3.2, "roi_high": 4.0, "percentile_75": 3.6},
        },
        "social": {
            "oncology": {"roi_low": 1.2, "roi_mid": 1.8, "roi_high": 2.4, "percentile_75": 2.1},
            "vaccines":  {"roi_low": 1.6, "roi_mid": 2.2, "roi_high": 2.8, "percentile_75": 2.5},
            "pharma":    {"roi_low": 1.4, "roi_mid": 2.0, "roi_high": 2.6, "percentile_75": 2.3},
        },
        "display": {
            "oncology": {"roi_low": 0.7, "roi_mid": 1.1, "roi_high": 1.5, "percentile_75": 1.3},
            "vaccines":  {"roi_low": 0.9, "roi_mid": 1.3, "roi_high": 1.7, "percentile_75": 1.5},
            "pharma":    {"roi_low": 0.8, "roi_mid": 1.2, "roi_high": 1.6, "percentile_75": 1.4},
        },
        "audio": {
            "oncology": {"roi_low": 0.8, "roi_mid": 1.2, "roi_high": 1.6, "percentile_75": 1.4},
            "vaccines":  {"roi_low": 1.0, "roi_mid": 1.5, "roi_high": 2.0, "percentile_75": 1.7},
            "pharma":    {"roi_low": 0.9, "roi_mid": 1.3, "roi_high": 1.7, "percentile_75": 1.5},
        },
    },

    # ── Engagement benchmarks (pharma industry public data) ────────
    "engagement": {
        "sfmc":   {"open_rate": 0.3465, "ctr": 0.028,  "unsubscribe_rate": 0.0025, "source": "phamax Digital 2024"},
        "nexgen": {"open_rate": 0.3465, "ctr": 0.028,  "unsubscribe_rate": 0.0025, "source": "phamax Digital 2024"},
        "display":     {"ctr": 0.0035, "source": "phamax Digital 2024"},
        "paid_search": {"ctr": 0.065,  "source": "phamax Digital 2024"},
    },
}


# ─── Single Channel Benchmark ─────────────────────────────────────
def benchmark_fetcher(channel: str, sub_vertical: str, current_roi: float = None) -> dict:
    """
    current_roi is optional. When omitted the function returns the industry
    benchmark ranges without a brand-specific comparison. The synthesizer is
    responsible for cross-referencing the actual ROI from mmm_retriever results.
    """
    channel      = channel.lower().strip()
    sub_vertical = sub_vertical.lower().strip()

    channel_benchmarks = BENCHMARKS["channels"].get(channel)
    if not channel_benchmarks:
        return {
            "status":  "not_found",
            "message": f"No benchmark data available for channel: {channel}",
            "channel": channel,
        }

    vertical_benchmarks = channel_benchmarks.get(sub_vertical)
    if not vertical_benchmarks:
        return {
            "status":       "not_found",
            "message":      f"No benchmark data for {channel} in {sub_vertical}",
            "channel":      channel,
            "sub_vertical": sub_vertical,
        }

    roi_low  = vertical_benchmarks["roi_low"]
    roi_mid  = vertical_benchmarks["roi_mid"]
    roi_high = vertical_benchmarks["roi_high"]
    p75      = vertical_benchmarks["percentile_75"]

    engagement = BENCHMARKS.get("engagement", {}).get(channel, {})

    # Base result always includes the benchmark ranges so the synthesizer can
    # compare against whatever actual ROI was retrieved by mmm_retriever.
    result = {
        "status":                "found",
        "channel":               channel,
        "sub_vertical":          sub_vertical,
        "benchmark_roi_low":     roi_low,
        "benchmark_roi_mid":     roi_mid,
        "benchmark_roi_high":    roi_high,
        "benchmark_p75":         p75,
        "engagement_benchmarks": engagement,
        "data_source":           BENCHMARKS["source"],
        "note": (
            "Use the actual ROI from mmm_retriever results to determine the "
            "brand's position relative to these benchmark ranges."
        ),
    }

    if current_roi is None or current_roi <= 0:
        # Return ranges only — synthesizer will do the comparison
        result["current_roi"] = None
        result["insight"] = (
            f"{channel.title()} industry benchmarks for {sub_vertical}: "
            f"low={roi_low}x, median={roi_mid}x, 75th pct={p75}x, high={roi_high}x. "
            f"Compare the brand's actual ROI (from retrieved MMM data) against these ranges."
        )
        return result

    # percentile position when current_roi is known
    if current_roi >= roi_high:
        percentile_position = "top quartile (>75th percentile)"
        performance_label   = "outperformer"
    elif current_roi >= p75:
        percentile_position = "75th percentile"
        performance_label   = "strong performer"
    elif current_roi >= roi_mid:
        percentile_position = "median to 75th percentile"
        performance_label   = "average performer"
    elif current_roi >= roi_low:
        percentile_position = "below median"
        performance_label   = "underperformer"
    else:
        percentile_position = "bottom quartile (<25th percentile)"
        performance_label   = "significant underperformer"

    gap_to_median = round(roi_mid - current_roi, 2)
    gap_to_p75    = round(p75 - current_roi, 2)

    if current_roi >= roi_mid:
        insight = (
            f"{channel.title()} ROI of {current_roi}x is above the industry median "
            f"of {roi_mid}x for {sub_vertical} brands. "
            f"This channel is performing well — consider protecting or growing this budget."
        )
    else:
        insight = (
            f"{channel.title()} ROI of {current_roi}x is below the industry median "
            f"of {roi_mid}x for {sub_vertical} brands (gap: {abs(gap_to_median):.2f}x). "
            f"Investigate creative quality, targeting, or frequency before increasing spend."
        )

    result.update({
        "current_roi":         current_roi,
        "gap_to_median":       gap_to_median,
        "gap_to_p75":          gap_to_p75,
        "percentile_position": percentile_position,
        "performance_label":   performance_label,
        "insight":             insight,
    })
    return result


# ─── Batch — compare all channels in one MMM output ───────────────
def benchmark_all_channels(channels: list, sub_vertical: str) -> list:
    """
    channels: list of {"name": str} or {"name": str, "roi": float}.
    roi is optional — omit it when the actual ROI should come from mmm_retriever.
    """
    return [
        benchmark_fetcher(
            channel=ch["name"],
            sub_vertical=sub_vertical,
            current_roi=ch.get("roi"),
        )
        for ch in channels
    ]


# ─── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    # single channel test
    result = benchmark_fetcher("doximity", "oncology", 2.4)
    print(json.dumps(result, indent=2))

    print("\n" + "=" * 60)

    # batch test
    sample_channels = [
        {"name": "doximity",          "roi": 2.4},
        {"name": "salesforce_calls",  "roi": 1.6},
        {"name": "paid_search",       "roi": 4.1},
        {"name": "tv",                "roi": 1.1},
        {"name": "display",           "roi": 0.9},
    ]
    batch = benchmark_all_channels(sample_channels, "oncology")
    for b in batch:
        if b["status"] == "found":
            print(f"\n{b['channel'].upper()} — {b['performance_label'].title()}")
            print(f"  Current ROI : {b['current_roi']}x  |  Benchmark mid: {b['benchmark_roi_mid']}x")
            print(f"  Position    : {b['percentile_position']}")
            print(f"  Insight     : {b['insight']}")