"""
Data downloader for the Amazon ESCI dataset.
Downloads, samples, silver-labels, and splits product data for NER training.
Rule-based silver labels use structured fields (brand, category) and curated dictionaries.
"""

import os
import json
import random
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import yaml
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Curated color vocabulary (200 entries)
COLORS = [
    "red", "blue", "green", "yellow", "orange", "purple", "pink", "black",
    "white", "gray", "grey", "brown", "beige", "ivory", "cream", "tan",
    "navy", "teal", "turquoise", "cyan", "magenta", "violet", "indigo",
    "maroon", "crimson", "scarlet", "coral", "salmon", "peach", "lavender",
    "lilac", "mauve", "rose", "ruby", "amber", "gold", "silver", "bronze",
    "copper", "olive", "khaki", "charcoal", "slate", "platinum", "champagne",
    "caramel", "chocolate", "coffee", "espresso", "mocha", "mint", "sage",
    "forest", "emerald", "jade", "lime", "lemon", "mustard", "saffron",
    "ochre", "rust", "burgundy", "wine", "plum", "eggplant", "fuchsia",
    "hot pink", "baby blue", "sky blue", "royal blue", "cobalt", "sapphire",
    "aqua", "seafoam", "grass green", "hunter green", "pine", "dark green",
    "light green", "neon green", "electric blue", "neon pink", "neon yellow",
    "off white", "snow white", "bone", "ecru", "pearl", "nude", "blush",
    "dusty rose", "dusty blue", "dusty purple", "pastel pink", "pastel blue",
    "pastel green", "pastel yellow", "pastel purple", "pastel orange",
    "multicolor", "multi-color", "multicolored", "tie-dye", "ombre",
    "gradient", "printed", "striped", "plaid", "checkered", "floral",
    "camouflage", "camo", "leopard", "zebra", "snake", "animal print",
    "heather gray", "heather grey", "heather blue", "heather green",
    "heather purple", "heather navy", "heather charcoal", "melange",
    "denim", "acid wash", "stone wash", "distressed", "washed",
    "dark", "light", "medium", "bright", "deep", "pale", "muted",
    "vibrant", "vivid", "rich", "bold", "soft", "warm", "cool",
    "natural", "classic", "vintage", "retro", "transparent", "clear",
    "translucent", "opaque", "glitter", "metallic", "glossy", "matte",
    "shiny", "sparkle", "iridescent", "holographic", "neon", "fluorescent",
    "dark blue", "dark red", "dark green", "dark brown", "dark gray",
    "light blue", "light red", "light green", "light brown", "light gray",
    "bright red", "bright blue", "bright green", "bright yellow", "bright orange",
    "deep red", "deep blue", "deep green", "deep purple", "deep navy",
    "pale blue", "pale green", "pale yellow", "pale pink", "pale purple",
    "warm white", "cool white", "warm gray", "cool gray", "steel blue",
    "midnight blue", "ocean blue", "ice blue", "powder blue", "periwinkle",
    "dusty pink", "hot coral", "terracotta", "clay", "sand", "desert",
    "cinnamon", "nutmeg", "walnut", "mahogany", "chestnut", "auburn",
    "strawberry", "watermelon", "tomato", "pumpkin", "tangerine", "mango",
    "banana", "lemon yellow", "buttercup", "sunshine", "golden", "wheat",
    "oatmeal", "linen", "vanilla", "almond", "hazel", "driftwood",
]

# Curated material vocabulary (150 entries)
MATERIALS = [
    "cotton", "polyester", "nylon", "wool", "silk", "linen", "leather",
    "suede", "velvet", "denim", "canvas", "fleece", "spandex", "lycra",
    "elastane", "rayon", "viscose", "acrylic", "cashmere", "angora",
    "mohair", "alpaca", "merino", "bamboo", "hemp", "lace", "chiffon",
    "satin", "taffeta", "organza", "tulle", "tweed", "flannel", "corduroy",
    "twill", "jersey", "knit", "woven", "mesh", "net", "crochet",
    "embroidered", "printed", "jacquard", "brocade", "sequin", "beaded",
    "metallic fabric", "faux leather", "vegan leather", "pu leather",
    "pvc", "rubber", "latex", "foam", "memory foam", "gel", "silicone",
    "plastic", "abs plastic", "polypropylene", "polycarbonate", "acetal",
    "neoprene", "gore-tex", "softshell", "hardshell", "ripstop",
    "microfiber", "microsuede", "sherpa", "plush", "terry", "waffle",
    "carbon fiber", "fiberglass", "kevlar", "ballistic nylon",
    "stainless steel", "aluminum", "titanium", "brass", "copper", "zinc",
    "iron", "steel", "chrome", "nickel", "silver", "gold", "platinum",
    "wood", "oak", "pine", "walnut", "maple", "bamboo wood", "teak",
    "mahogany", "birch", "cedar", "plywood", "mdf", "particle board",
    "glass", "tempered glass", "crystal", "acrylic glass", "plexiglass",
    "ceramic", "porcelain", "stoneware", "earthenware", "terracotta",
    "stone", "marble", "granite", "slate", "quartz", "concrete",
    "natural rubber", "synthetic rubber", "thermoplastic", "resin",
    "epoxy", "fibre", "fiber", "down", "feather", "fill", "padding",
    "batting", "stuffing", "recycled", "organic", "sustainable", "eco",
    "blend", "mixed media", "composite", "laminate", "coated",
    "water resistant", "waterproof", "breathable", "moisture wicking",
    "quick dry", "uv resistant", "antimicrobial", "odor resistant",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(config: dict) -> pd.DataFrame:
    """Download Amazon ESCI or fallback dataset and return as DataFrame."""
    try:
        logger.info("Attempting to load amazon_esci dataset...")
        ds = load_dataset("tasksource/amazon-esci", split="train")
        df = ds.to_pandas()
        logger.info(f"Loaded amazon_esci: {len(df)} rows")
        # Normalize column names to expected fields
        col_map = {}
        if "product_title" in df.columns:
            col_map["product_title"] = "title"
        if "product_brand" in df.columns:
            col_map["product_brand"] = "brand"
        if "product_description" in df.columns:
            col_map["product_description"] = "description"
        if "product_bullet_point" in df.columns:
            col_map["product_bullet_point"] = "bullet_points"
        df = df.rename(columns=col_map)
        return df
    except Exception as e:
        logger.warning(f"amazon_esci load failed: {e}")

    try:
        logger.info("Falling back to McAuley-Lab Amazon Reviews 2023...")
        ds = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            "raw_meta_Clothing_Shoes_and_Jewelry",
            split="full",
            trust_remote_code=True,
        )
        df = ds.to_pandas()
        logger.info(f"Loaded Amazon Reviews 2023: {len(df)} rows")
        col_map = {}
        if "title" not in df.columns and "store" in df.columns:
            df["title"] = df.get("title", df.get("description", ""))
        if "brand" not in df.columns and "store" in df.columns:
            df["brand"] = df["store"]
        if "category" not in df.columns and "main_category" in df.columns:
            df["category"] = df["main_category"]
        return df
    except Exception as e:
        logger.error(f"Fallback dataset load also failed: {e}")
        raise RuntimeError("Could not load any dataset. Check internet connection.") from e


def sample_balanced(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    """Sample proportionally across top-20 categories."""
    cat_col = "category" if "category" in df.columns else (
        "product_type" if "product_type" in df.columns else None
    )
    if cat_col is None or df[cat_col].isna().all():
        return df.sample(min(sample_size, len(df)), random_state=seed).reset_index(drop=True)

    top_cats = df[cat_col].value_counts().head(20).index
    df_top = df[df[cat_col].isin(top_cats)].copy()
    if len(df_top) < sample_size:
        df_top = df.copy()

    per_cat = sample_size // min(20, df_top[cat_col].nunique())
    parts = []
    for cat in df_top[cat_col].unique():
        chunk = df_top[df_top[cat_col] == cat]
        parts.append(chunk.sample(min(per_cat, len(chunk)), random_state=seed))

    sampled = pd.concat(parts).sample(frac=1, random_state=seed)
    if len(sampled) < sample_size:
        remaining = df[~df.index.isin(sampled.index)]
        extra = remaining.sample(min(sample_size - len(sampled), len(remaining)), random_state=seed)
        sampled = pd.concat([sampled, extra])

    return sampled.head(sample_size).reset_index(drop=True)


def _find_span(tokens: list[str], phrase: str) -> list[tuple[int, int]]:
    """Find all token spans matching a multi-word phrase (case-insensitive)."""
    phrase_tokens = phrase.lower().split()
    n = len(phrase_tokens)
    spans = []
    for i in range(len(tokens) - n + 1):
        if [t.lower() for t in tokens[i:i+n]] == phrase_tokens:
            spans.append((i, i + n))
    return spans


def generate_silver_labels(row: pd.Series) -> dict | None:
    """
    Rule-based silver label generation.
    Uses brand/category fields and color/material dictionaries to produce BIO tags.
    Returns None if the title is missing or too short.
    """
    title = str(row.get("title", "") or "").strip()
    if len(title) < 3:
        return None

    tokens = title.split()
    labels = ["O"] * len(tokens)

    def tag_span(span_start: int, span_end: int, entity_type: str):
        labels[span_start] = f"B-{entity_type}"
        for i in range(span_start + 1, span_end):
            labels[i] = f"I-{entity_type}"

    # Brand labels from structured field
    brand = str(row.get("brand", "") or "").strip()
    if brand and brand.lower() not in ("", "nan", "none", "unknown"):
        for start, end in _find_span(tokens, brand):
            tag_span(start, end, "BRAND")

    # Category labels from structured field
    category = str(row.get("category", row.get("product_type", "")) or "").strip()
    if category and category.lower() not in ("", "nan", "none", "unknown"):
        cat_short = category.split(">")[-1].strip() if ">" in category else category
        for start, end in _find_span(tokens, cat_short):
            tag_span(start, end, "CATEGORY")

    # Color labels from dictionary
    title_lower = title.lower()
    for color in sorted(COLORS, key=len, reverse=True):
        if color in title_lower:
            for start, end in _find_span(tokens, color):
                if labels[start] == "O":
                    tag_span(start, end, "COLOR")

    # Material labels from dictionary
    for material in sorted(MATERIALS, key=len, reverse=True):
        if material in title_lower:
            for start, end in _find_span(tokens, material):
                if labels[start] == "O":
                    tag_span(start, end, "MATERIAL")

    return {"tokens": tokens, "labels": labels}


def split_and_save(records: list[dict], processed_dir: str, config: dict):
    """Split into train/val/test and save as JSONL."""
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
    for split_name, split_records in splits.items():
        path = os.path.join(processed_dir, f"{split_name}.json")
        with open(path, "w") as f:
            for rec in split_records:
                f.write(json.dumps(rec) + "\n")
        logger.info(f"Saved {len(split_records)} records to {path}")


def main(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    raw_dir = config["paths"]["raw_data"]
    processed_dir = config["paths"]["processed_data"]
    samples_dir = config["paths"]["samples"]
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    # Download and sample
    df = download_dataset(config)
    df = sample_balanced(df, config["data"]["sample_size"], config["data"]["seed"])
    logger.info(f"Sampled {len(df)} products")

    # Save raw sample
    raw_path = os.path.join(raw_dir, "products_sampled.parquet")
    df.to_parquet(raw_path, index=False)
    logger.info(f"Saved raw sample to {raw_path}")

    # Save 500-row sample CSV
    sample_path = os.path.join(samples_dir, "products_sample.csv")
    df.head(500).to_csv(sample_path, index=False)
    logger.info(f"Saved 500-row sample to {sample_path}")

    # Generate silver labels
    logger.info("Generating silver labels...")
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Silver labeling"):
        result = generate_silver_labels(row)
        if result is not None and len(result["tokens"]) > 0:
            records.append(result)

    logger.info(f"Generated {len(records)} labeled examples")

    # Split and save
    split_and_save(records, processed_dir, config)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
