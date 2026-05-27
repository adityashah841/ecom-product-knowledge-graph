"""
Text cleaning, tokenization with label alignment, and matching pair generation.
"""

import re
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_product_text(text: str, max_tokens: int = 128) -> str:
    """Lowercase, strip HTML, normalize whitespace, remove non-ASCII, truncate."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML tags
    text = re.sub(r"[^\x00-\x7F]+", " ", text)    # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()[:max_tokens]
    return " ".join(tokens)


def tokenize_and_align_labels(examples: dict, tokenizer, label2id: dict, max_length: int = 128) -> dict:
    """
    Align word-level BIO labels to wordpiece tokens.
    First subword of each word keeps the label; continuation subwords get -100.
    """
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    all_labels = []
    for i, word_labels in enumerate(examples["labels"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                tag = word_labels[word_id] if word_id < len(word_labels) else "O"
                label_ids.append(label2id.get(tag, 0))
            else:
                # Continuation subword — mask with -100 so loss ignores it
                label_ids.append(-100)
            prev_word_id = word_id
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized


def build_matching_pairs(products_df: pd.DataFrame, output_path: str, seed: int = 42, max_pairs: int = 50000):
    """
    Build positive and hard-negative pairs for bi-encoder training.

    Positive pairs: same brand AND same category (likely duplicate/variant).
    Hard negatives: same category, different brand (structurally similar but distinct).
    """
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    title_col = "title" if "title" in products_df.columns else "product_title"
    brand_col = "brand" if "brand" in products_df.columns else "product_brand"
    cat_col = "category" if "category" in products_df.columns else (
        "product_type" if "product_type" in products_df.columns else None
    )

    df = products_df[[title_col, brand_col] + ([cat_col] if cat_col else [])].copy()
    df.columns = ["title", "brand"] + (["category"] if cat_col else [])
    df["title"] = df["title"].fillna("").astype(str).str.strip()
    df["brand"] = df["brand"].fillna("unknown").astype(str).str.lower().str.strip()
    if "category" not in df.columns:
        df["category"] = "unknown"
    df["category"] = df["category"].fillna("unknown").astype(str).str.lower().str.strip()
    df = df[df["title"].str.len() > 5].reset_index(drop=True)

    pairs = []

    # Positive pairs: same brand + same category
    grouped = df.groupby(["brand", "category"])
    for (brand, cat), group in grouped:
        if brand in ("unknown", "nan", "") or len(group) < 2:
            continue
        idxs = group.index.tolist()
        n_pos = min(5, len(idxs) * (len(idxs) - 1) // 2)
        sample_idxs = rng.choice(idxs, size=min(n_pos * 2, len(idxs)), replace=False)
        for j in range(0, len(sample_idxs) - 1, 2):
            pairs.append({
                "text_a": df.loc[sample_idxs[j], "title"],
                "text_b": df.loc[sample_idxs[j + 1], "title"],
                "label": 1,
            })
            if len(pairs) >= max_pairs // 2:
                break
        if len(pairs) >= max_pairs // 2:
            break

    # Hard negative pairs: same category, different brand
    cat_grouped = df.groupby("category")
    for cat, group in cat_grouped:
        brands = group["brand"].unique()
        if len(brands) < 2:
            continue
        b1, b2 = rng.choice(brands, size=2, replace=False)
        g1 = group[group["brand"] == b1]
        g2 = group[group["brand"] == b2]
        if len(g1) == 0 or len(g2) == 0:
            continue
        idx_a = rng.choice(g1.index)
        idx_b = rng.choice(g2.index)
        pairs.append({
            "text_a": df.loc[idx_a, "title"],
            "text_b": df.loc[idx_b, "title"],
            "label": 0,
        })
        if len(pairs) >= max_pairs:
            break

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")

    pos = sum(1 for p in pairs if p["label"] == 1)
    neg = sum(1 for p in pairs if p["label"] == 0)
    logger.info(f"Saved {len(pairs)} matching pairs ({pos} positive, {neg} negative) to {output_path}")
    return pairs
