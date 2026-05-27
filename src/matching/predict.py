"""
Bi-encoder inference: find duplicate product listings using FAISS.
"""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_texts(texts: List[str], model, tokenizer, device: torch.device, batch_size: int = 64) -> np.ndarray:
    """Encode a list of texts to L2-normalized embeddings."""
    model.eval()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            with autocast():
                embeddings = model.encode(enc["input_ids"], enc["attention_mask"])
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings).astype("float32")


def find_duplicates(
    product_list: List[str],
    model,
    tokenizer,
    device: torch.device,
    threshold: float = 0.85,
    batch_size: int = 64,
) -> List[Tuple[int, int, float]]:
    """
    Find pairs of products above the similarity threshold using FAISS flat index.
    Returns list of (idx_a, idx_b, similarity_score).
    """
    import faiss

    embeddings = encode_texts(product_list, model, tokenizer, device, batch_size)
    n, d = embeddings.shape

    index = faiss.IndexFlatIP(d)  # Inner product on normalized vectors = cosine similarity
    index.add(embeddings)

    k = min(10, n)
    distances, indices = index.search(embeddings, k)

    duplicates = []
    seen = set()
    for i in range(n):
        for j_rank in range(1, k):
            j = indices[i][j_rank]
            if j <= i:
                continue
            sim = float(distances[i][j_rank])
            if sim >= threshold:
                key = (i, j)
                if key not in seen:
                    seen.add(key)
                    duplicates.append((i, j, round(sim, 4)))

    logger.info(f"Found {len(duplicates)} duplicate pairs above threshold {threshold}")
    return sorted(duplicates, key=lambda x: -x[2])
