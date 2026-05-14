import os
import json
import logging
from tqdm import tqdm
from .config import DATA_FOLDER
from .db import get_collection

logger = logging.getLogger(__name__)


def chunk_mmm_file(filepath: str) -> list[dict]:
    with open(filepath) as f:
        data = json.load(f)

    meta     = data["metadata"]
    summary  = data["model_summary"]
    fin      = data["financials"]
    brand    = meta["brand"]
    vertical = meta["sub_vertical"]
    year     = meta["modelling_period"]["end"].split("-")[0]
    model_id = data["model_id"]

    chunks = []

    summary_text = (
        f"{brand.title()} is a {vertical} brand. "
        f"Total revenue is ${fin['nrv']:,}. "
        f"The MMM model covers {meta['modelling_period']['start']} to {meta['modelling_period']['end']}. "
        f"Base contribution is {summary['base_contribution_pct']*100:.0f}% "
        f"and incremental contribution is {summary['incremental_contribution_pct']*100:.0f}%. "
        f"Model R-squared is {summary['r_squared']} with MAPE of {summary['mape']}."
    )
    chunks.append({
        "text": summary_text,
        "metadata": {
            "type": "summary", "brand": brand, "sub_vertical": vertical,
            "year": year, "model_id": model_id, "channel": "all", "category": "all",
        },
        "id": f"{model_id}_summary",
    })

    for ch in data["channels"]:
        channel_text = (
            f"{ch['name'].title()} is a {ch['category'].upper()} channel "
            f"for {brand.title()} ({vertical}). "
            f"Revenue contribution was ${ch['revenue_contribution']:,}. "
            f"Spend was ${ch['spend']:,} with an ROI of {ch['roi']}x. "
            f"Reach was {ch['reach']:,} with frequency of {ch['frequency']}. "
            f"Adstock decay rate is {ch['transformation']['adstock_decay_rate']}. "
            f"Tactics used: {', '.join(ch['tactics'])}."
        )
        if ch.get("calls_delivered"):
            channel_text += f" Calls delivered: {ch['calls_delivered']:,}."

        chunks.append({
            "text": channel_text,
            "metadata": {
                "type": "channel", "brand": brand, "sub_vertical": vertical,
                "year": year, "model_id": model_id,
                "channel": ch["name"], "category": ch["category"],
            },
            "id": f"{model_id}_{ch['name']}",
        })

    return chunks


def ingest_all(folder_path: str = DATA_FOLDER, collection=None) -> None:
    if collection is None:
        collection = get_collection()

    all_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    existing_ids = set(collection.get()["ids"])

    for i, filename in enumerate(tqdm(all_files, desc="Ingesting"), 1):
        filepath = os.path.join(folder_path, filename)
        chunks = [c for c in chunk_mmm_file(filepath) if c["id"] not in existing_ids]

        if not chunks:
            logger.debug("Skipping %s — already ingested", filename)
            continue

        collection.add(
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
            ids=[c["id"] for c in chunks],
        )
        logger.info("[%d/%d] Ingested %s — %d chunks", i, len(all_files), filename, len(chunks))

    logger.info("Done. Total chunks in collection: %d", collection.count())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest_all()
