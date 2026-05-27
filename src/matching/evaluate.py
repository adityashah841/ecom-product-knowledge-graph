"""
Bi-encoder evaluation: threshold sweep for precision/recall/F1 tradeoff.
"""

import json
import logging
import os

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score

from src.matching.dataset import MatchingDataset, matching_collate_fn
from src.matching.model import BiEncoderModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_at_threshold(similarities: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (similarities >= threshold).astype(int)
    return {
        "threshold": round(threshold, 2),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall": round(recall_score(labels, preds, zero_division=0), 4),
        "f1": round(f1_score(labels, preds, zero_division=0), 4),
    }


def run_evaluation(config_path: str, model_path: str, results_path: str, device_str: str = "cuda"):
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    m_cfg = config["matching"]
    paths = config["paths"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    dataset = MatchingDataset(os.path.join(paths["processed_data"], "matching_pairs.json"), m_cfg["model_name"])
    n = len(dataset)
    _, val_ds = torch.utils.data.random_split(dataset, [int(n * 0.8), n - int(n * 0.8)])
    val_loader = DataLoader(val_ds, batch_size=m_cfg["batch_size"], shuffle=False, collate_fn=matching_collate_fn, num_workers=0)

    model = BiEncoderModel(m_cfg["model_name"]).to(device)
    model.load(model_path, device)
    model.eval()

    all_sims, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                similarity, _, _ = model(
                    batch["input_ids_a"], batch["attention_mask_a"],
                    batch["input_ids_b"], batch["attention_mask_b"],
                )
            all_sims.extend(similarity.cpu().numpy().tolist())
            all_labels.extend(batch["label"].cpu().numpy().tolist())

    sims = np.array(all_sims)
    labels = np.array(all_labels, dtype=int)

    thresholds = np.arange(0.70, 0.96, 0.05)
    sweep = [evaluate_at_threshold(sims, labels, t) for t in thresholds]

    default_result = evaluate_at_threshold(sims, labels, m_cfg["similarity_threshold"])
    results = {
        "threshold_sweep": sweep,
        "at_configured_threshold": default_result,
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Matching results saved to {results_path}")
    logger.info(f"F1 @ {m_cfg['similarity_threshold']}: {default_result['f1']}")
    return results
