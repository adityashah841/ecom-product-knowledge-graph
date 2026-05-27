"""
Bi-encoder fine-tuning with in-batch negatives (MultipleNegativesRankingLoss)
plus explicit BCE for labeled pairs.
"""

import os
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

from src.matching.dataset import MatchingDataset, matching_collate_fn
from src.matching.model import BiEncoderModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def mnrl_loss(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    """
    Multiple Negatives Ranking Loss.
    For each anchor in emb_a, the paired emb_b is the positive; all others are in-batch negatives.
    """
    scores = torch.mm(emb_a, emb_b.T)  # (B, B)
    labels = torch.arange(scores.size(0), device=scores.device)
    return F.cross_entropy(scores * 20.0, labels)  # scale=20 is common for cosine similarity


def train(config_path: str, output_dir: str, device_str: str):
    config = load_config(config_path)
    m_cfg = config["matching"]
    paths = config["paths"]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and device_str == "cuda":
        logger.warning("CUDA not available — falling back to CPU")

    pairs_path = os.path.join(paths["processed_data"], "matching_pairs.json")
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Matching pairs not found at {pairs_path}. Run data pipeline first.")

    dataset = MatchingDataset(pairs_path, m_cfg["model_name"])
    n = len(dataset)
    train_size = int(n * 0.8)
    val_size = n - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=m_cfg["batch_size"], shuffle=True, collate_fn=matching_collate_fn, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=m_cfg["batch_size"], shuffle=False, collate_fn=matching_collate_fn, num_workers=0)

    model = BiEncoderModel(m_cfg["model_name"]).to(device)
    optimizer = AdamW(model.parameters(), lr=m_cfg["learning_rate"])
    total_steps = len(train_loader) * m_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_steps)
    scaler = GradScaler(enabled=m_cfg["fp16"] and torch.cuda.is_available())
    bce_loss = nn.BCEWithLogitsLoss()

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    torch.backends.cudnn.benchmark = True

    for epoch in range(m_cfg["num_epochs"]):
        model.train()
        torch.cuda.empty_cache()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            if m_cfg["fp16"] and torch.cuda.is_available():
                with autocast():
                    similarity, emb_a, emb_b = model(
                        batch["input_ids_a"], batch["attention_mask_a"],
                        batch["input_ids_b"], batch["attention_mask_b"],
                    )
                    loss_rank = mnrl_loss(emb_a, emb_b)
                    loss_bce = bce_loss(similarity, batch["label"])
                    loss = 0.5 * loss_rank + 0.5 * loss_bce
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                similarity, emb_a, emb_b = model(
                    batch["input_ids_a"], batch["attention_mask_a"],
                    batch["input_ids_b"], batch["attention_mask_b"],
                )
                loss_rank = mnrl_loss(emb_a, emb_b)
                loss_bce = bce_loss(similarity, batch["label"])
                loss = 0.5 * loss_rank + 0.5 * loss_bce
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    similarity, emb_a, emb_b = model(
                        batch["input_ids_a"], batch["attention_mask_a"],
                        batch["input_ids_b"], batch["attention_mask_b"],
                    )
                    loss = bce_loss(similarity, batch["label"])
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save(os.path.join(output_dir, "biencoder_best.pt"))
            logger.info("  Saved best bi-encoder checkpoint")

    logger.info("Bi-encoder training complete")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output_dir", default="models/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    train(args.config, args.output_dir, args.device)
