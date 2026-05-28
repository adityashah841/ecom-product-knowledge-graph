"""
Data loader for the Amazon ESCI product catalogue (real dataset).

Reads shopping_queries_dataset_products.parquet directly from the cloned
amazon-science/esci-data repository — no HuggingFace loaders required.

Columns used:
  product_id, product_title, product_brand, product_color, product_locale

Silver labels are generated via rule-based brand/color/category matching.
CATEGORY uses a curated vocabulary of common Amazon product type terms
matched as whole words in the title (same pattern as COLOR).
Expected silver accuracy: ~85-92% (intentional noise on real-world titles).
"""

import json
import logging
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Color vocabulary ──────────────────────────────────────────────────────────
COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "beige", "ivory", "cream", "tan",
    "navy", "teal", "turquoise", "cyan", "magenta", "violet", "indigo",
    "maroon", "crimson", "scarlet", "coral", "salmon", "peach", "lavender",
    "lilac", "mauve", "rose", "ruby", "amber", "gold", "silver", "bronze",
    "copper", "olive", "khaki", "charcoal", "slate", "platinum", "champagne",
    "caramel", "chocolate", "coffee", "espresso", "mocha", "mint", "sage",
    "forest", "emerald", "jade", "lime", "lemon", "mustard", "saffron",
    "ochre", "rust", "burgundy", "wine", "plum", "fuchsia", "aqua",
    "seafoam", "cobalt", "sapphire", "periwinkle", "terracotta", "clay",
    "sand", "blush", "dusty rose", "hot pink", "sky blue", "royal blue",
    "dark green", "light blue", "midnight blue", "rose gold", "off white",
    "heather gray", "heather grey", "stone", "natural", "nude", "bone",
    "multicolor", "multi-color", "multicolored",
]

_COLOR_PATTERNS = {
    c: re.compile(r"\b" + re.escape(c) + r"\b", re.IGNORECASE)
    for c in sorted(COLORS, key=len, reverse=True)
}

# ── Category vocabulary ───────────────────────────────────────────────────────
CATEGORIES = [
    # Apparel
    "shirt", "t-shirt", "tshirt", "polo", "blouse", "top", "tank top",
    "hoodie", "sweatshirt", "sweater", "cardigan", "jacket", "coat", "vest",
    "pants", "jeans", "shorts", "leggings", "skirt", "dress", "suit",
    "socks", "underwear", "bra", "pajamas", "swimsuit", "swimwear",
    # Footwear
    "shoes", "sneakers", "boots", "sandals", "slippers", "loafers", "heels",
    "running shoes", "hiking boots", "flip flops",
    # Electronics
    "laptop", "tablet", "monitor", "keyboard", "mouse", "headphones",
    "earbuds", "speaker", "microphone", "webcam", "router", "charger",
    "cable", "adapter", "battery", "hard drive", "ssd", "flash drive",
    "printer", "scanner", "projector", "camera", "tripod",
    "phone case", "screen protector",
    # Home & Kitchen
    "blender", "toaster", "coffee maker", "kettle", "air fryer",
    "slow cooker", "rice cooker", "pan", "pot", "skillet", "baking sheet",
    "knife", "cutting board", "mixing bowl", "colander",
    "vacuum cleaner", "air purifier", "humidifier", "fan", "heater",
    "lamp", "light bulb", "curtains", "rug", "pillow", "blanket",
    "mattress", "sheets", "towel", "shower curtain",
    # Tools & Hardware
    "drill", "screwdriver", "hammer", "wrench", "pliers", "tape measure",
    "level", "saw", "power tool", "extension cord",
    # Beauty & Health
    "shampoo", "conditioner", "moisturizer", "sunscreen", "lipstick",
    "mascara", "foundation", "perfume", "deodorant", "toothbrush",
    "razor", "trimmer", "hair dryer", "straightener", "curling iron",
    # Sports & Outdoors
    "yoga mat", "dumbbell", "resistance band", "foam roller",
    "tent", "sleeping bag", "backpack", "water bottle", "helmet",
    "gloves", "knee pad", "ankle brace",
    # Office & Stationery
    "notebook", "pen", "pencil", "marker", "stapler", "folder", "binder",
    "desk organizer", "calendar", "planner",
    # Baby & Kids
    "diaper", "baby monitor", "stroller", "car seat", "crib",
    "baby bottle", "pacifier", "toy", "puzzle",
    # Pet Supplies
    "dog food", "cat food", "dog bed", "cat bed", "leash", "collar",
    "pet carrier", "litter box",
]

_CATEGORY_PATTERNS = {
    c: re.compile(r"\b" + re.escape(c) + r"\b", re.IGNORECASE)
    for c in sorted(CATEGORIES, key=len, reverse=True)
}


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _find_esci_products_parquet() -> str:
    """Search common locations for the ESCI products parquet file."""
    desktop = Path.home() / "Desktop"
    candidates = [
        desktop / "esci-data" / "shopping_queries_dataset" / "shopping_queries_dataset_products.parquet",
        Path("../esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"),
    ]
    for p in candidates:
        if p.exists():
            return str(p.resolve())
    raise FileNotFoundError(
        "ESCI products parquet not found. Expected at:\n"
        f"  {candidates[0]}\n"
        "Clone and pull LFS:\n"
        "  git clone https://github.com/amazon-science/esci-data.git\n"
        "  git -C esci-data lfs pull --include='shopping_queries_dataset/shopping_queries_dataset_products.parquet'"
    )


# ── ESCI loader ───────────────────────────────────────────────────────────────

def load_esci_products(filepath: str, sample_size: int, seed: int) -> pd.DataFrame:
    """
    Load ESCI products parquet, filter to US locale, sample proportionally
    by product_type_id (top-20 types), return [product_id, title, brand, color].
    """
    logger.info(f"Loading ESCI products from {filepath} ...")
    df = pd.read_parquet(filepath)
    logger.info(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    # Filter to English US
    if "product_locale" in df.columns:
        df = df[df["product_locale"] == "us"].copy()
        logger.info(f"After US filter: {len(df):,} rows")

    # Normalize columns
    col_map = {
        "product_title": "title",
        "product_brand": "brand",
        "product_color": "color",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    for col in ["title", "brand", "color", "product_id"]:
        if col not in df.columns:
            df[col] = ""

    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["brand"] = df["brand"].fillna("").astype(str).str.strip()
    df["color"] = df["color"].fillna("").astype(str).str.strip()

    # Drop titles that are too short
    df = df[df["title"].str.len() >= 5].reset_index(drop=True)
    logger.info(f"After title-length filter: {len(df):,} rows")

    # Proportional sampling across top-20 product_type_id values
    type_col = "product_type_id" if "product_type_id" in df.columns else None
    if not type_col:
        logger.warning(
            "product_type_id column not found in ESCI parquet — falling back to random sampling. "
            "Proportional-by-type sampling is unavailable."
        )
    if type_col:
        top_types = df[type_col].value_counts().head(20).index
        df_pool = df[df[type_col].isin(top_types)].copy()
        if len(df_pool) < sample_size:
            df_pool = df.copy()
        counts = df_pool[type_col].value_counts()
        per_type = max(1, sample_size // len(counts))
        parts = [
            grp.sample(min(per_type, len(grp)), random_state=seed)
            for _, grp in df_pool.groupby(type_col)
        ]
        sampled = pd.concat(parts)
        if len(sampled) < sample_size:
            remaining = df_pool[~df_pool.index.isin(sampled.index)]
            extra = min(sample_size - len(sampled), len(remaining))
            if extra > 0:
                sampled = pd.concat([sampled, remaining.sample(extra, random_state=seed)])
        sampled = sampled.sample(frac=1, random_state=seed).head(sample_size)
    else:
        sampled = df.sample(min(sample_size, len(df)), random_state=seed)

    sampled = sampled.reset_index(drop=True)
    logger.info(f"Final sample: {len(sampled):,} products")
    return sampled[["product_id", "title", "brand", "color"]]


# ── Silver label generator ────────────────────────────────────────────────────

def _find_span(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    phrase_tokens = phrase.lower().split()
    n = len(phrase_tokens)
    spans = []
    for i in range(len(tokens) - n + 1):
        if [t.lower() for t in tokens[i:i + n]] == phrase_tokens:
            spans.append((i, i + n))
    return spans


def generate_silver_labels(row: pd.Series) -> dict | None:
    """
    Rule-based BIO labeling for real ESCI product titles.

    Brand rules:
      - Brand must be >= 3 characters (avoids single-letter false matches)
      - Skip if brand appears > 3 times (boilerplate indicator)
    Color rules:
      - Whole-word regex to avoid partial matches (e.g. "red" in "hundred")
      - Uses product_color field first, then vocabulary scan
    Category rules:
      - Whole-word regex scan against curated CATEGORIES vocabulary
      - Longest match wins (vocabulary sorted by length descending)
      - Only tagged if span is not already occupied by BRAND or COLOR
    """
    title = str(row.get("title", "") or "").strip()
    if len(title) < 5:
        return None

    tokens = title.split()
    labels = ["O"] * len(tokens)

    def tag_span(start: int, end: int, entity_type: str):
        labels[start] = f"B-{entity_type}"
        for i in range(start + 1, end):
            labels[i] = f"I-{entity_type}"

    # Brand
    brand = str(row.get("brand", "") or "").strip()
    if len(brand) >= 3:
        title_lower = title.lower()
        if title_lower.count(brand.lower()) <= 3:
            for start, end in _find_span(tokens, brand):
                if labels[start] == "O":
                    tag_span(start, end, "BRAND")
                    break

    # Color from product_color field
    product_color = str(row.get("color", "") or "").strip()
    if len(product_color) >= 3:
        for start, end in _find_span(tokens, product_color):
            if labels[start] == "O":
                tag_span(start, end, "COLOR")
                break

    # Color from vocabulary (whole-word regex)
    title_lower = title.lower()
    for color, pattern in _COLOR_PATTERNS.items():
        if pattern.search(title_lower):
            for start, end in _find_span(tokens, color):
                if all(labels[i] == "O" for i in range(start, end)):
                    tag_span(start, end, "COLOR")
                    break

    # Category from vocabulary (whole-word regex, longest match first)
    for category, pattern in _CATEGORY_PATTERNS.items():
        if pattern.search(title_lower):
            for start, end in _find_span(tokens, category):
                if all(labels[i] == "O" for i in range(start, end)):
                    tag_span(start, end, "CATEGORY")

    return {"tokens": tokens, "labels": labels}


# ── Split and save ────────────────────────────────────────────────────────────

def split_and_save(records: list, processed_dir: str, config: dict):
    seed = config["data"]["seed"]
    random.seed(seed)
    random.shuffle(records)
    n = len(records)
    train_end = int(n * config["data"]["train_split"])
    val_end = train_end + int(n * config["data"]["val_split"])
    splits = {
        "ner_train": records[:train_end],
        "ner_val": records[train_end:val_end],
        "ner_test": records[val_end:],
    }
    os.makedirs(processed_dir, exist_ok=True)
    for name, recs in splits.items():
        path = os.path.join(processed_dir, f"{name}.json")
        with open(path, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {len(recs):,} records → {path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    samples_dir = config["paths"]["samples"]
    results_dir = config["paths"]["results"]
    for d in [raw_dir, processed_dir, samples_dir, results_dir]:
        os.makedirs(d, exist_ok=True)

    esci_path = _find_esci_products_parquet()
    logger.info(f"Using ESCI file: {esci_path}")

    df = load_esci_products(esci_path, config["data"]["sample_size"], config["data"]["seed"])

    raw_path = os.path.join(raw_dir, "products_sampled.parquet")
    df.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw sample → {raw_path}")

    sample_path = os.path.join(samples_dir, "products_sample.csv")
    df.head(500).to_csv(sample_path, index=False)
    logger.info(f"Saved 500-row sample → {sample_path}")

    # Silver labeling
    logger.info("Generating silver labels...")
    records, labeled = [], 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Silver labeling"):
        result = generate_silver_labels(row)
        if result and len(result["tokens"]) > 0:
            records.append(result)
            if any(l != "O" for l in result["labels"]):
                labeled += 1

    coverage = labeled / max(len(records), 1) * 100
    logger.info(f"Silver label coverage: {coverage:.1f}% ({labeled:,}/{len(records):,})")

    stats = {
        "total_examples": len(records),
        "labeled_examples": labeled,
        "coverage_pct": round(coverage, 2),
        "unlabeled_examples": len(records) - labeled,
    }
    with open(os.path.join(results_dir, "silver_label_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    split_and_save(records, processed_dir, config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
