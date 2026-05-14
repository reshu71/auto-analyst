import io
import json
import base64
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

matplotlib.use("Agg")   # non-interactive backend — safe for server use

# ─── Color Palette ────────────────────────────────────────────────
COLORS = {
    "primary":    "#2563EB",
    "teal":       "#0D9488",
    "orange":     "#EA580C",
    "purple":     "#7C3AED",
    "red":        "#DC2626",
    "green":      "#16A34A",
    "gray":       "#94A3B8",
    "dark":       "#0F172A",
    "light_gray": "#F1F5F9",
    "hcp":        "#2563EB",
    "consumer":   "#0D9488",
}

HCP_CHANNELS      = {"doximity", "medscape", "pulsepoint", "deepintent", "sfmc", "nexgen", "salesforce_calls"}
CONSUMER_CHANNELS = {"tv", "streaming_tv", "online_video", "paid_search", "social", "display", "audio"}


def _channel_color(channel_name: str) -> str:
    return COLORS["hcp"] if channel_name in HCP_CHANNELS else COLORS["consumer"]


def _to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _base_style(fig, ax, title: str):
    fig.patch.set_facecolor(COLORS["dark"])
    ax.set_facecolor(COLORS["dark"])
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")


# ─── 1. ROI Bar Chart ─────────────────────────────────────────────
def roi_bar_chart(channels: list, brand: str = "", sub_vertical: str = "") -> str:
    """
    Horizontal bar chart of ROI per channel.

    Args:
        channels: list of dicts with keys: name, roi, category
        brand:    brand name for the title
        sub_vertical: sub_vertical for the title
    """
    channels_sorted = sorted(channels, key=lambda x: x["roi"], reverse=True)
    names  = [ch["name"].replace("_", " ").title() for ch in channels_sorted]
    rois   = [ch["roi"] for ch in channels_sorted]
    colors = [_channel_color(ch["name"]) for ch in channels_sorted]

    fig, ax = plt.subplots(figsize=(9, max(4, len(names) * 0.55)))
    bars = ax.barh(names, rois, color=colors, edgecolor="none", height=0.6)

    # value labels
    for bar, val in zip(bars, rois):
        ax.text(
            val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}x", va="center", ha="left",
            color="white", fontsize=9
        )

    title = f"Channel ROI — {brand.title()} ({sub_vertical})" if brand else "Channel ROI"
    _base_style(fig, ax, title)
    ax.set_xlabel("ROI (Revenue per $1 Spent)", color="white", fontsize=10)
    ax.invert_yaxis()

    # legend
    hcp_patch      = mpatches.Patch(color=COLORS["hcp"],      label="HCP")
    consumer_patch = mpatches.Patch(color=COLORS["consumer"], label="Consumer")
    ax.legend(handles=[hcp_patch, consumer_patch], facecolor="#1E293B",
              labelcolor="white", fontsize=9, loc="lower right")

    plt.tight_layout()
    return _to_base64(fig)


# ─── 2. Spend vs Revenue Contribution Scatter ─────────────────────
def spend_vs_revenue_chart(channels: list, brand: str = "") -> str:
    """
    Scatter plot of spend vs revenue contribution.
    Bubble size = ROI. Diagonal = break-even (ROI = 1).
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    _base_style(fig, ax, f"Spend vs Revenue Contribution — {brand.title()}" if brand else "Spend vs Revenue")

    for ch in channels:
        color  = _channel_color(ch["name"])
        size   = ch["roi"] * 80
        ax.scatter(ch["spend"], ch["revenue_contribution"],
                   s=size, color=color, alpha=0.8, edgecolors="white", linewidths=0.5)
        ax.annotate(
            ch["name"].replace("_", " ").title(),
            (ch["spend"], ch["revenue_contribution"]),
            textcoords="offset points", xytext=(6, 4),
            color="white", fontsize=7.5
        )

    # break-even line (ROI = 1)
    max_val = max(max(ch["spend"] for ch in channels), max(ch["revenue_contribution"] for ch in channels))
    ax.plot([0, max_val], [0, max_val], color=COLORS["gray"],
            linestyle="--", linewidth=1, label="Break-even (ROI=1x)", alpha=0.6)

    ax.set_xlabel("Spend ($)", color="white", fontsize=10)
    ax.set_ylabel("Revenue Contribution ($)", color="white", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)

    plt.tight_layout()
    return _to_base64(fig)


# ─── 3. Revenue Decomposition Waterfall ───────────────────────────
def revenue_waterfall_chart(model_summary: dict, channels: list, brand: str = "") -> str:
    """
    Waterfall chart showing base + each channel's contribution to total revenue.

    Args:
        model_summary: dict with total_revenue, base_contribution_pct
        channels:      list of channel dicts with revenue_contribution
    """
    total_revenue = model_summary["total_revenue"]
    base_pct      = model_summary["base_contribution_pct"]
    base_revenue  = total_revenue * base_pct

    # sort channels by contribution descending, take top 8
    top_channels = sorted(channels, key=lambda x: x["revenue_contribution"], reverse=True)[:8]

    labels  = ["Base"] + [ch["name"].replace("_", " ").title() for ch in top_channels] + ["Total"]
    values  = [base_revenue] + [ch["revenue_contribution"] for ch in top_channels] + [0]
    colors  = [COLORS["purple"]] + [_channel_color(ch["name"]) for ch in top_channels] + [COLORS["green"]]

    # cumulative for waterfall positioning
    running = 0
    bottoms = []
    for i, v in enumerate(values[:-1]):
        bottoms.append(running)
        running += v
    bottoms.append(0)   # total bar starts at 0
    values[-1] = running

    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.1), 6))
    bars = ax.bar(labels, values, bottom=bottoms, color=colors, edgecolor="none", width=0.6)

    # value labels on bars
    for bar, val, bot in zip(bars, values, bottoms):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bot + val / 2,
            f"${val/1e6:.1f}M",
            ha="center", va="center", color="white", fontsize=8, fontweight="bold"
        )

    _base_style(fig, ax, f"Revenue Decomposition — {brand.title()}" if brand else "Revenue Decomposition")
    ax.set_ylabel("Revenue ($)", color="white", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    return _to_base64(fig)


# ─── 4. Scenario Before/After Chart ──────────────────────────────
def scenario_bar_chart(scenario_results: list) -> str:
    """
    Grouped bar chart comparing current vs new spend and revenue
    after a scenario simulation.

    Args:
        scenario_results: list of dicts from scenario_simulator()
    """
    channels = [r["channel"].replace("_", " ").title() for r in scenario_results]
    current_rev = [r["current_revenue"] for r in scenario_results]
    new_rev     = [r["new_revenue"]     for r in scenario_results]

    x     = np.arange(len(channels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(channels) * 1.4), 6))
    bars1 = ax.bar(x - width / 2, current_rev, width, label="Current Revenue",
                   color=COLORS["primary"], alpha=0.85, edgecolor="none")
    bars2 = ax.bar(x + width / 2, new_rev,     width, label="Projected Revenue",
                   color=COLORS["teal"],    alpha=0.85, edgecolor="none")

    # delta labels
    for i, r in enumerate(scenario_results):
        delta = r["revenue_change_pct"]
        color = COLORS["green"] if delta >= 0 else COLORS["red"]
        ax.text(i, max(current_rev[i], new_rev[i]) + max(current_rev) * 0.02,
                f"{delta:+.1f}%", ha="center", color=color, fontsize=9, fontweight="bold")

    _base_style(fig, ax, "Scenario Simulation — Revenue Before vs After")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Revenue ($)", color="white", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)

    plt.tight_layout()
    return _to_base64(fig)


# ─── 5. Benchmark Comparison Chart ───────────────────────────────
def benchmark_comparison_chart(benchmark_results: list, sub_vertical: str = "") -> str:
    """
    Chart comparing current ROI vs industry benchmark mid and p75.

    Args:
        benchmark_results: list of dicts from benchmark_fetcher()
        sub_vertical:      for the title
    """
    found = [b for b in benchmark_results if b["status"] == "found"]
    if not found:
        return ""

    found_sorted = sorted(found, key=lambda x: x["current_roi"], reverse=True)
    channels  = [b["channel"].replace("_", " ").title() for b in found_sorted]
    current   = [b["current_roi"]        for b in found_sorted]
    bench_mid = [b["benchmark_roi_mid"]  for b in found_sorted]
    bench_p75 = [b["benchmark_p75"]      for b in found_sorted]

    x     = np.arange(len(channels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(9, len(channels) * 1.3), 6))

    ax.bar(x - width,     current,   width, label="Your ROI",         color=COLORS["primary"], alpha=0.9, edgecolor="none")
    ax.bar(x,             bench_mid, width, label="Industry Median",   color=COLORS["gray"],   alpha=0.7, edgecolor="none")
    ax.bar(x + width,     bench_p75, width, label="Industry 75th Pct", color=COLORS["orange"], alpha=0.7, edgecolor="none")

    # above/below median markers
    for i, b in enumerate(found_sorted):
        color = COLORS["green"] if b["current_roi"] >= b["benchmark_roi_mid"] else COLORS["red"]
        marker = "▲" if b["current_roi"] >= b["benchmark_roi_mid"] else "▼"
        ax.text(x[i] - width, b["current_roi"] + 0.05, marker,
                ha="center", color=color, fontsize=10)

    _base_style(fig, ax, f"ROI vs Industry Benchmarks — {sub_vertical.title()}" if sub_vertical else "ROI vs Industry Benchmarks")
    ax.set_xticks(x)
    ax.set_xticklabels(channels, rotation=20, ha="right", color="white", fontsize=9)
    ax.set_ylabel("ROI (x)", color="white", fontsize=10)
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)

    plt.tight_layout()
    return _to_base64(fig)


# ─── 6. Power Curve Shape Chart ──────────────────────────────────
def power_curve_chart(channel: str, current_spend: float, current_revenue: float) -> str:
    """
    Shows the diminishing returns curve for a channel.
    Marks current spend position on the curve.
    """
    from .scenario_simulator import POWER_CURVES

    exponent = POWER_CURVES.get(channel.lower(), 0.5)

    # generate curve from 10% to 250% of current spend
    spend_range  = np.linspace(current_spend * 0.1, current_spend * 2.5, 300)
    revenue_curve = current_revenue * (spend_range / current_spend) ** exponent

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(spend_range, revenue_curve, color=COLORS["primary"], linewidth=2.5, label="Response Curve")
    ax.scatter([current_spend], [current_revenue],
               color=COLORS["orange"], s=120, zorder=5, label="Current Position", edgecolors="white")

    # shade region beyond current spend to show diminishing returns
    beyond_mask = spend_range >= current_spend
    ax.fill_between(spend_range[beyond_mask], revenue_curve[beyond_mask],
                    alpha=0.1, color=COLORS["red"], label="Diminishing return zone")

    _base_style(fig, ax, f"Response Curve — {channel.replace('_', ' ').title()} (exponent: {exponent})")
    ax.set_xlabel("Spend ($)", color="white", fontsize=10)
    ax.set_ylabel("Revenue Contribution ($)", color="white", fontsize=10)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
    ax.legend(facecolor="#1E293B", labelcolor="white", fontsize=9)

    plt.tight_layout()
    return _to_base64(fig)


# ─── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    os.makedirs("chart_outputs", exist_ok=True)

    # sample data
    sample_channels = [
        {"name": "salesforce_calls", "spend": 3_500_000, "roi": 1.8, "revenue_contribution": 6_300_000, "category": "hcp"},
        {"name": "doximity",         "spend": 800_000,   "roi": 2.9, "revenue_contribution": 2_320_000, "category": "hcp"},
        {"name": "medscape",         "spend": 600_000,   "roi": 2.6, "revenue_contribution": 1_560_000, "category": "hcp"},
        {"name": "tv",               "spend": 2_000_000, "roi": 1.3, "revenue_contribution": 2_600_000, "category": "consumer"},
        {"name": "paid_search",      "spend": 400_000,   "roi": 3.4, "revenue_contribution": 1_360_000, "category": "consumer"},
        {"name": "display",          "spend": 250_000,   "roi": 1.0, "revenue_contribution": 250_000,   "category": "consumer"},
    ]

    sample_summary = {
        "total_revenue": 20_000_000,
        "base_contribution_pct": 0.62,
    }

    # chart 1 — ROI bar
    b64 = roi_bar_chart(sample_channels, brand="keytruda", sub_vertical="oncology")
    with open("chart_outputs/roi_bar.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("✓ roi_bar.png")

    # chart 2 — spend vs revenue scatter
    b64 = spend_vs_revenue_chart(sample_channels, brand="keytruda")
    with open("chart_outputs/spend_vs_revenue.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("✓ spend_vs_revenue.png")

    # chart 3 — waterfall
    b64 = revenue_waterfall_chart(sample_summary, sample_channels, brand="keytruda")
    with open("chart_outputs/waterfall.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("✓ waterfall.png")

    # chart 4 — scenario before/after
    from scenario_simulator import scenario_simulator
    scenario_results = [
        scenario_simulator({"channel": ch["name"], "current_spend": ch["spend"],
                            "current_revenue_contribution": ch["revenue_contribution"],
                            "budget_change_pct": -0.20 if ch["name"] == "tv" else 0.20 if ch["name"] == "paid_search" else 0.0})
        for ch in sample_channels
    ]
    b64 = scenario_bar_chart(scenario_results)
    with open("chart_outputs/scenario.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("✓ scenario.png")

    # chart 5 — benchmark comparison
    from benchmark_fetcher import benchmark_all_channels
    bench_results = benchmark_all_channels(sample_channels, "oncology")
    b64 = benchmark_comparison_chart(bench_results, sub_vertical="oncology")
    with open("chart_outputs/benchmark.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("✓ benchmark.png")

    # chart 6 — power curve
    b64 = power_curve_chart("tv", current_spend=2_000_000, current_revenue=2_600_000)
    with open("chart_outputs/power_curve.png", "wb") as f:
        f.write(base64.b64decode(b64))
    print("✓ power_curve.png")

    print("\nAll charts saved to chart_outputs/")