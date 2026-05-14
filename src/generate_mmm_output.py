import json
import numpy as np
import os
import random
from .config import DATA_FOLDER

# ─── Configuration ────────────────────────────────────────────────
BRANDS = {
    "oncology": ["keytruda", "welireg", "lenvima", "lynparza", "qliftara"],
    "vaccines": ["gardasil", "vaxelis", "vaxneuvance", "proquad", "rotateq", "vaqta", "capvaxive"],
    "pharma":   ["bridion", "belsomra", "verquovo", "januvia", "dificid", "zerbaxia", "del-pif"],
}

HCP_VENDOR_TACTICS = {
    "doximity":   ["emails", "edetails", "alerts", "banners"],
    "medscape":   ["emails", "edetails", "alerts", "banners"],
    "pulsepoint": ["banners"],
    "deepintent": ["banners"],
    "sfmc":       ["hq_emails"],
    "nexgen":     ["emails"],
}

CONSUMER_CHANNEL_TACTICS = {
    "tv":           ["linear_spots"],
    "streaming_tv": ["ctv", "ott"],
    "online_video": ["pre_roll", "mid_roll"],
    "paid_search":  ["brand_search", "non_brand_search"],
    "social":       ["facebook", "instagram"],
    "display":      ["banners"],
    "audio":        ["streaming_audio"],
}

CYCLES = [
    {"mod": ("2021-01", "2023-12"), "rep": ("2023-01", "2023-12")},
    {"mod": ("2022-01", "2024-12"), "rep": ("2024-01", "2024-12")},
    {"mod": ("2023-01", "2025-12"), "rep": ("2025-01", "2025-12")},
]

VENDOR_ROI = {
    "doximity":        {"oncology": (3.2, 0.6), "vaccines": (2.4, 0.5), "pharma": (2.8, 0.5)},
    "medscape":        {"oncology": (3.0, 0.6), "vaccines": (2.2, 0.5), "pharma": (2.6, 0.5)},
    "pulsepoint":      {"oncology": (1.5, 0.3), "vaccines": (1.3, 0.3), "pharma": (1.4, 0.3)},
    "deepintent":      {"oncology": (1.6, 0.3), "vaccines": (1.4, 0.3), "pharma": (1.5, 0.3)},
    "sfmc":            {"oncology": (2.0, 0.4), "vaccines": (1.8, 0.4), "pharma": (1.9, 0.4)},
    "nexgen":          {"oncology": (2.1, 0.4), "vaccines": (1.9, 0.4), "pharma": (2.0, 0.4)},
    "salesforce_calls":{"oncology": (1.8, 0.4), "vaccines": (2.0, 0.5), "pharma": (1.6, 0.4)},
    "tv":              {"oncology": (1.3, 0.3), "vaccines": (2.5, 0.6), "pharma": (1.5, 0.3)},
    "streaming_tv":    {"oncology": (1.5, 0.3), "vaccines": (2.7, 0.5), "pharma": (1.7, 0.3)},
    "online_video":    {"oncology": (1.6, 0.3), "vaccines": (2.4, 0.5), "pharma": (1.8, 0.3)},
    "paid_search":     {"oncology": (3.5, 0.7), "vaccines": (3.0, 0.6), "pharma": (3.2, 0.6)},
    "social":          {"oncology": (1.8, 0.4), "vaccines": (2.2, 0.5), "pharma": (2.0, 0.4)},
    "display":         {"oncology": (1.1, 0.2), "vaccines": (1.3, 0.3), "pharma": (1.2, 0.2)},
    "audio":           {"oncology": (1.2, 0.3), "vaccines": (1.5, 0.3), "pharma": (1.3, 0.3)},
}

SPEND_RANGES = {
    "doximity":        (300_000,  1_200_000),
    "medscape":        (300_000,  1_200_000),
    "pulsepoint":      (100_000,    500_000),
    "deepintent":      (100_000,    500_000),
    "sfmc":            (50_000,     300_000),
    "nexgen":          (50_000,     300_000),
    "salesforce_calls":(2_000_000, 8_000_000),
    "tv":              (2_000_000, 10_000_000),
    "streaming_tv":    (1_000_000,  5_000_000),
    "online_video":    (500_000,   3_000_000),
    "paid_search":     (300_000,   2_000_000),
    "social":          (500_000,   3_000_000),
    "display":         (200_000,   1_000_000),
    "audio":           (200_000,     800_000),
}

ADSTOCK_RANGES = {
    "doximity":        (0.3, 0.6),
    "medscape":        (0.3, 0.6),
    "pulsepoint":      (0.2, 0.4),
    "deepintent":      (0.2, 0.4),
    "sfmc":            (0.3, 0.5),
    "nexgen":          (0.3, 0.5),
    "salesforce_calls":(0.5, 0.8),
    "tv":              (0.3, 0.6),
    "streaming_tv":    (0.2, 0.5),
    "online_video":    (0.2, 0.4),
    "paid_search":     (0.1, 0.3),
    "social":          (0.2, 0.4),
    "display":         (0.1, 0.3),
    "audio":           (0.2, 0.4),
}


def sample_roi(channel: str, sub_vertical: str) -> float:
    mean, std = VENDOR_ROI[channel][sub_vertical]
    return max(0.1, round(np.random.normal(mean, std), 2))


def build_channel(name: str, category: str, sub_vertical: str, tactics: list, calls_delivered=None) -> dict:
    spend = random.randint(*SPEND_RANGES[name])
    roi   = sample_roi(name, sub_vertical)
    decay = round(random.uniform(*ADSTOCK_RANGES[name]), 2)
    reach = (
        random.randint(5_000, 30_000)
        if category == "hcp"
        else random.randint(500_000, 8_000_000)
    )
    return {
        "name": name,
        "category": category,
        "tactics": tactics,
        "spend": spend,
        "calls_delivered": calls_delivered,
        "reach": reach,
        "frequency": round(
            random.uniform(2.0, 5.0) if category == "hcp" else random.uniform(1.5, 4.5), 1
        ),
        "transformation": {"adstock_decay_rate": decay},
        "coefficient": round(random.uniform(0.001, 0.01), 4),
        "roi": roi,
        "revenue_contribution": int(spend * roi),
    }


def generate_channels(sub_vertical: str) -> list[dict]:
    channels = [
        build_channel(
            name="salesforce_calls",
            category="hcp",
            sub_vertical=sub_vertical,
            tactics=["primary_care_calls", "specialty_calls"],
            calls_delivered=random.randint(20_000, 80_000),
        )
    ]

    for vendor in random.sample(list(HCP_VENDOR_TACTICS.keys()), k=random.randint(2, 4)):
        channels.append(build_channel(
            name=vendor, category="hcp", sub_vertical=sub_vertical,
            tactics=HCP_VENDOR_TACTICS[vendor],
        ))

    for chan in random.sample(list(CONSUMER_CHANNEL_TACTICS.keys()), k=random.randint(2, 4)):
        channels.append(build_channel(
            name=chan, category="consumer", sub_vertical=sub_vertical,
            tactics=CONSUMER_CHANNEL_TACTICS[chan],
        ))

    return channels


def create_mock_data(count: int = 50, output_dir: str = DATA_FOLDER) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for i in range(count):
        sub_vertical = random.choice(list(BRANDS.keys()))
        brand        = random.choice(BRANDS[sub_vertical])
        cycle        = random.choice(CYCLES)
        year_suffix  = cycle["mod"][1].split("-")[0]
        channels     = generate_channels(sub_vertical)

        total_incremental = sum(c["revenue_contribution"] for c in channels)
        base_pct          = round(random.uniform(0.50, 0.70), 2)
        incremental_pct   = round(1 - base_pct, 2)
        total_revenue     = int(total_incremental / incremental_pct)

        data = {
            "model_id": f"pharma_{sub_vertical}_{brand}_{year_suffix}_{i}",
            "metadata": {
                "industry": "pharma",
                "sub_vertical": sub_vertical,
                "brand": brand,
                "modelling_period": {"start": cycle["mod"][0], "end": cycle["mod"][1]},
                "reporting_period": {"start": cycle["rep"][0], "end": cycle["rep"][1]},
            },
            "financials": {
                "nrv": total_revenue,
                "pgm": int(total_revenue * random.uniform(0.60, 0.80)),
            },
            "model_summary": {
                "total_revenue": total_revenue,
                "base_contribution_pct": base_pct,
                "incremental_contribution_pct": incremental_pct,
                "r_squared": round(random.uniform(0.85, 0.96), 2),
                "mape": round(random.uniform(0.04, 0.12), 2),
            },
            "channels": channels,
        }

        filename = f"{sub_vertical}_{brand}_{year_suffix}_{i}.json"
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(data, f, indent=2)

        print(f"[{i+1}/{count}] {filename}")


if __name__ == "__main__":
    create_mock_data(50)
    print(f"\nDone — 50 MMM output files in /{DATA_FOLDER}")
