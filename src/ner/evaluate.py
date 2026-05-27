"""NER evaluation using seqeval for entity-level precision, recall, F1."""

import json
import logging
import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def decode_predictions(logits: torch.Tensor, labels: torch.Tensor, id2label: dict) -> Tuple[list, list]:
    """Convert logit tensors to seqeval-compatible string label lists."""
    predictions = torch.argmax(logits, dim=-1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    pred_seqs, true_seqs = [], []
    for pred_row, label_row in zip(predictions, labels_np):
        pred_seq, true_seq = [], []
        for p, l in zip(pred_row, label_row):
            if l == -100:
                continue
            pred_seq.append(id2label.get(p, "O"))
            true_seq.append(id2label.get(l, "O"))
        pred_seqs.append(pred_seq)
        true_seqs.append(true_seq)

    return pred_seqs, true_seqs


def evaluate_model(model, data_loader: DataLoader, id2label: dict, label_list: list, device: torch.device) -> Tuple[float, float]:
    """Run evaluation loop; return (f1, loss)."""
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
            total_loss += outputs.loss.item()
            preds, trues = decode_predictions(outputs.logits, batch["labels"], id2label)
            all_preds.extend(preds)
            all_true.extend(trues)

    f1 = f1_score(all_true, all_preds)
    avg_loss = total_loss / max(len(data_loader), 1)
    return f1, avg_loss


def evaluate_and_save(model, test_loader: DataLoader, id2label: dict, label_list: list, device: torch.device, results_path: str):
    """Full evaluation with per-entity breakdown; saves results JSON."""
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                outputs = model(**batch)
            preds, trues = decode_predictions(outputs.logits, batch["labels"], id2label)
            all_preds.extend(preds)
            all_true.extend(trues)

    report = classification_report(all_true, all_preds, output_dict=True)
    overall_f1 = f1_score(all_true, all_preds)
    overall_prec = precision_score(all_true, all_preds)
    overall_rec = recall_score(all_true, all_preds)

    entity_types = ["BRAND", "PRODUCT", "CATEGORY", "ATTRIBUTE", "COLOR", "MATERIAL"]
    per_entity = {}
    for etype in entity_types:
        if etype in report:
            per_entity[etype] = {
                "precision": round(report[etype]["precision"], 4),
                "recall": round(report[etype]["recall"], 4),
                "f1": round(report[etype]["f1-score"], 4),
            }
        else:
            per_entity[etype] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    results = {
        "overall_f1": round(overall_f1, 4),
        "overall_precision": round(overall_prec, 4),
        "overall_recall": round(overall_rec, 4),
        "per_entity": per_entity,
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"NER evaluation results saved to {results_path}")
    logger.info(f"Overall F1: {overall_f1:.4f}")
    return results
