import json

# ─── Power Curve Exponents ────────────────────────────────────────
# Represents diminishing returns per channel
# Lower exponent = stronger diminishing returns
# Higher exponent = more linear response
# Source: calibrated from pharma MMM model outputs

POWER_CURVES = {
    # HCP digital — moderate diminishing returns
    "doximity":         0.5,
    "medscape":         0.5,
    "pulsepoint":       0.6,
    "deepintent":       0.6,
    "sfmc":             0.5,
    "nexgen":           0.5,
    # Salesforce — strongest diminishing returns (high frequency = low marginal gain)
    "salesforce_calls": 0.3,
    # Consumer — TV has heavy diminishing returns, paid search is near linear
    "tv":               0.4,
    "streaming_tv":     0.4,
    "online_video":     0.5,
    "paid_search":      0.8,   # most linear — intent-driven, scales efficiently
    "social":           0.6,
    "display":          0.6,
    "audio":            0.6,
}

DEFAULT_EXPONENT = 0.5


# ─── Single Channel Scenario ──────────────────────────────────────
def scenario_simulator(input_json: dict) -> dict:
    """
    Simulates the revenue impact of a budget change on a single channel.

    Input:
        input_json = {
            "channel":                   "tv",
            "current_spend":             3_000_000,
            "current_revenue_contribution": 4_200_000,
            "budget_change_pct":         -0.20   # -20% cut, +0.10 = 10% increase
        }

    Returns a full before/after comparison dict with plain English insight.
    """
    channel           = input_json["channel"].lower().strip()
    current_spend     = input_json["current_spend"]
    current_revenue   = input_json["current_revenue_contribution"]
    budget_change_pct = input_json["budget_change_pct"]

    power_exponent = POWER_CURVES.get(channel, DEFAULT_EXPONENT)

    # core simulation
    new_spend   = current_spend * (1 + budget_change_pct)
    new_revenue = current_revenue * (new_spend / current_spend) ** power_exponent

    # derived metrics
    revenue_change     = new_revenue - current_revenue
    revenue_change_pct = (revenue_change / current_revenue) * 100
    current_roi        = round(current_revenue / current_spend, 2)
    new_roi            = round(new_revenue / new_spend, 2)
    roi_change         = round(new_roi - current_roi, 2)
    spend_change_pct   = budget_change_pct * 100

    # ── Plain English efficiency note ─────────────────────────────
    direction        = "cut" if budget_change_pct < 0 else "increase"
    revenue_dir      = "loss" if revenue_change < 0 else "gain"

    efficiency_note = (
        f"A {abs(spend_change_pct):.0f}% spend {direction} on {channel} "
        f"leads to a {abs(revenue_change_pct):.1f}% revenue {revenue_dir} "
        f"(power curve exponent: {power_exponent}). "
        f"ROI moves from {current_roi}x to {new_roi}x. "
    )

    # add diminishing returns context
    if power_exponent <= 0.4 and budget_change_pct < 0:
        efficiency_note += (
            f"Due to strong diminishing returns (exponent {power_exponent}), "
            f"the revenue loss is proportionally smaller than the spend cut — "
            f"this is a relatively efficient channel to reduce."
        )
    elif power_exponent <= 0.4 and budget_change_pct > 0:
        efficiency_note += (
            f"Due to strong diminishing returns (exponent {power_exponent}), "
            f"additional spend yields proportionally less revenue — "
            f"consider reallocating to higher-exponent channels like paid_search."
        )
    elif power_exponent >= 0.7 and budget_change_pct > 0:
        efficiency_note += (
            f"This channel has a near-linear response (exponent {power_exponent}) — "
            f"additional investment scales efficiently with minimal saturation."
        )
    elif power_exponent >= 0.7 and budget_change_pct < 0:
        efficiency_note += (
            f"This channel has a near-linear response (exponent {power_exponent}) — "
            f"spend cuts here will lead to proportionally large revenue losses."
        )
    else:
        efficiency_note += (
            f"Moderate diminishing returns (exponent {power_exponent}) — "
            f"budget changes have a balanced impact on revenue."
        )

    return {
        "channel":              channel,
        "power_exponent":       power_exponent,
        "current_spend":        round(current_spend, 2),
        "new_spend":            round(new_spend, 2),
        "spend_change_pct":     round(spend_change_pct, 1),
        "current_revenue":      round(current_revenue, 2),
        "new_revenue":          round(new_revenue, 2),
        "revenue_change":       round(revenue_change, 2),
        "revenue_change_pct":   round(revenue_change_pct, 2),
        "current_roi":          current_roi,
        "new_roi":              new_roi,
        "roi_change":           roi_change,
        "efficiency_note":      efficiency_note,
    }


# ─── Multi Channel Scenario ───────────────────────────────────────
def simulate_portfolio(channels: list, budget_changes: dict) -> dict:
    """
    Simulates budget changes across multiple channels simultaneously.

    Args:
        channels:       list of channel dicts from MMM JSON output
                        each dict must have: name, spend, revenue_contribution
        budget_changes: dict mapping channel name to budget_change_pct
                        e.g. {"tv": -0.20, "paid_search": 0.15}

    Returns:
        portfolio-level before/after summary + per-channel results
    """
    results         = []
    total_current_spend    = 0
    total_new_spend        = 0
    total_current_revenue  = 0
    total_new_revenue      = 0

    for ch in channels:
        name   = ch["name"].lower().strip()
        change = budget_changes.get(name, 0.0)   # default: no change

        result = scenario_simulator({
            "channel":                      name,
            "current_spend":                ch["spend"],
            "current_revenue_contribution": ch["revenue_contribution"],
            "budget_change_pct":            change,
        })
        results.append(result)

        total_current_spend   += ch["spend"]
        total_new_spend       += result["new_spend"]
        total_current_revenue += ch["revenue_contribution"]
        total_new_revenue     += result["new_revenue"]

    # portfolio-level metrics
    blended_current_roi = round(total_current_revenue / total_current_spend, 2) if total_current_spend else 0
    blended_new_roi     = round(total_new_revenue / total_new_spend, 2)         if total_new_spend     else 0
    portfolio_revenue_change     = total_new_revenue - total_current_revenue
    portfolio_revenue_change_pct = round((portfolio_revenue_change / total_current_revenue) * 100, 2) if total_current_revenue else 0

    return {
        "portfolio_summary": {
            "total_current_spend":         round(total_current_spend, 2),
            "total_new_spend":             round(total_new_spend, 2),
            "total_current_revenue":       round(total_current_revenue, 2),
            "total_new_revenue":           round(total_new_revenue, 2),
            "portfolio_revenue_change":    round(portfolio_revenue_change, 2),
            "portfolio_revenue_change_pct": portfolio_revenue_change_pct,
            "blended_current_roi":         blended_current_roi,
            "blended_new_roi":             blended_new_roi,
        },
        "channel_results": results,
    }


# ─── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":

    # test 1 — single channel
    print("=" * 60)
    print("SINGLE CHANNEL — TV -20%")
    result = scenario_simulator({
        "channel":                      "tv",
        "current_spend":                3_000_000,
        "current_revenue_contribution": 4_200_000,
        "budget_change_pct":            -0.20,
    })
    print(json.dumps(result, indent=2))

    # test 2 — paid search +30%
    print("\n" + "=" * 60)
    print("SINGLE CHANNEL — PAID SEARCH +30%")
    result2 = scenario_simulator({
        "channel":                      "paid_search",
        "current_spend":                500_000,
        "current_revenue_contribution": 1_600_000,
        "budget_change_pct":            0.30,
    })
    print(json.dumps(result2, indent=2))

    # test 3 — salesforce -10%
    print("\n" + "=" * 60)
    print("SINGLE CHANNEL — SALESFORCE -10%")
    result3 = scenario_simulator({
        "channel":                      "salesforce_calls",
        "current_spend":                4_000_000,
        "current_revenue_contribution": 6_400_000,
        "budget_change_pct":            -0.10,
    })
    print(json.dumps(result3, indent=2))

    # test 4 — portfolio scenario
    print("\n" + "=" * 60)
    print("PORTFOLIO — shift budget from TV to paid_search")
    sample_channels = [
        {"name": "tv",               "spend": 3_000_000, "revenue_contribution": 4_200_000},
        {"name": "paid_search",      "spend": 500_000,   "revenue_contribution": 1_600_000},
        {"name": "salesforce_calls", "spend": 4_000_000, "revenue_contribution": 6_400_000},
        {"name": "doximity",         "spend": 800_000,   "revenue_contribution": 2_200_000},
    ]
    budget_changes = {
        "tv":          -0.20,    # cut TV by 20%
        "paid_search": +0.40,    # reinvest into paid search
    }
    portfolio = simulate_portfolio(sample_channels, budget_changes)
    print("\nPortfolio Summary:")
    print(json.dumps(portfolio["portfolio_summary"], indent=2))
    print("\nChannel Results:")
    for ch in portfolio["channel_results"]:
        print(f"\n  {ch['channel'].upper()}")
        print(f"    Spend : ${ch['current_spend']:,.0f} → ${ch['new_spend']:,.0f}  ({ch['spend_change_pct']:+.0f}%)")
        print(f"    Revenue: ${ch['current_revenue']:,.0f} → ${ch['new_revenue']:,.0f}  ({ch['revenue_change_pct']:+.1f}%)")
        print(f"    ROI   : {ch['current_roi']}x → {ch['new_roi']}x")
        print(f"    Note  : {ch['efficiency_note']}")