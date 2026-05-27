"""
Semi-supervised pseudo-labeling loop for NER expansion.
Runs inference on unlabeled products, accepts high-confidence predictions,
and retrains the NER model iteratively.
"""

import json
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.cuda.amp import autocast
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def predict_with_confidence(model, tokenizer, tokens: list[str], id2label: dict, device: torch.device) -> tuple[list[str], float]:
    """
    Run NER inference on a tokenized example.
    Returns (labels, min_non_O_confidence) where min_non_O_confidence is the
    minimum confidence over all non-O predicted tokens.
    """
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        with autocast():
            outputs = model(**encoding)

    probs = F.softmax(outputs.logits[0], dim=-1)
    encoding_cpu = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=128)
    token_word_ids = encoding_cpu.word_ids()

    word_labels = {}
    word_confs = {}
    for token_idx, word_id in enumerate(token_word_ids):
        if word_id is None or word_id in word_labels:
            continue
        label_id = torch.argmax(probs[token_idx]).item()
        conf = probs[token_idx][label_id].item()
        word_labels[word_id] = id2label.get(label_id, "O")
        word_confs[word_id] = conf

    labels = [word_labels.get(i, "O") for i in range(len(tokens))]
    non_o_confs = [word_confs[i] for i in range(len(tokens)) if word_labels.get(i, "O") != "O"]
    min_conf = min(non_o_confs) if non_o_confs else 1.0

    return labels, min_conf


class PseudoLabelingLoop:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.ss_cfg = self.config["semi_supervised"]
        self.ner_cfg = self.config["ner"]
        self.paths = self.config["paths"]
        self.confidence_threshold = self.ss_cfg["confidence_threshold"]
        self.max_iterations = self.ss_cfg["max_iterations"]
        self.expansion_batch_size = self.ss_cfg["expansion_batch_size"]

    def load_unlabeled_pool(self, sampled_parquet: str, labeled_titles: set) -> list[dict]:
        """Load products not in any labeled split."""
        import pandas as pd
        df = pd.read_parquet(sampled_parquet)
        title_col = "title" if "title" in df.columns else "product_title"
        df = df[~df[title_col].isin(labeled_titles)].copy()
        df[title_col] = df[title_col].fillna("").astype(str).str.strip()
        df = df[df[title_col].str.len() > 5]
        return [{"tokens": row[title_col].split()} for _, row in df.iterrows()]

    def get_labeled_titles(self) -> set:
        titles = set()
        for split in ["ner_train", "ner_val", "ner_test"]:
            path = os.path.join(self.paths["processed_data"], f"{split}.json")
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for line in f:
                    rec = json.loads(line.strip())
                    titles.add(" ".join(rec["tokens"]))
        return titles

    def append_to_train(self, new_records: list[dict]):
        train_path = os.path.join(self.paths["processed_data"], "ner_train.json")
        with open(train_path, "a") as f:
            for rec in new_records:
                f.write(json.dumps(rec) + "\n")

    def run(self, model, tokenizer, id2label: dict, label_list: list, device: torch.device, output_dir: str):
        from src.ner.dataset import NERDataset, collate_fn
        from src.ner.evaluate import evaluate_model
        from src.ner.train import train_epoch, get_optimizer
        from torch.utils.data import DataLoader
        from torch.cuda.amp import GradScaler
        from transformers import get_linear_schedule_with_warmup

        label2id = {l: i for i, l in enumerate(label_list)}
        iteration_logs = []
        total_added = 0

        raw_parquet = os.path.join(self.paths["raw_data"], "products_sampled.parquet")
        if not os.path.exists(raw_parquet):
            logger.warning("Raw parquet not found — skipping pseudo-labeling")
            return

        labeled_titles = self.get_labeled_titles()
        unlabeled_pool = self.load_unlabeled_pool(raw_parquet, labeled_titles)
        logger.info(f"Unlabeled pool size: {len(unlabeled_pool)}")

        val_path = os.path.join(self.paths["processed_data"], "ner_val.json")
        val_dataset = NERDataset(val_path, tokenizer, label2id, self.config["data"]["max_seq_length"])
        val_loader = DataLoader(val_dataset, batch_size=self.ner_cfg["batch_size"], collate_fn=collate_fn, num_workers=0)

        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n=== Pseudo-labeling iteration {iteration}/{self.max_iterations} ===")

            # Measure val F1 before
            val_f1_before, _ = evaluate_model(model, val_loader, id2label, label_list, device)
            logger.info(f"Val F1 before iteration {iteration}: {val_f1_before:.4f}")

            # Collect pseudo-labeled examples
            model.eval()
            accepted = []
            batch_size = self.expansion_batch_size

            for i in tqdm(range(0, min(len(unlabeled_pool), 5000), 1), desc=f"Iter {iteration} pseudo-labeling"):
                example = unlabeled_pool[i]
                tokens = example["tokens"]
                if len(tokens) < 2:
                    continue
                try:
                    labels, min_conf = predict_with_confidence(model, tokenizer, tokens, id2label, device)
                    if min_conf >= self.confidence_threshold:
                        accepted.append({"tokens": tokens, "labels": labels})
                except Exception:
                    continue

            logger.info(f"Accepted {len(accepted)} pseudo-labeled examples")

            if not accepted:
                logger.info("No high-confidence examples found — stopping early")
                break

            # Remove accepted examples from the pool
            accepted_set = {" ".join(r["tokens"]) for r in accepted}
            unlabeled_pool = [e for e in unlabeled_pool if " ".join(e["tokens"]) not in accepted_set]

            # Append to training data and retrain
            self.append_to_train(accepted)
            total_added += len(accepted)

            train_path = os.path.join(self.paths["processed_data"], "ner_train.json")
            train_dataset = NERDataset(train_path, tokenizer, label2id, self.config["data"]["max_seq_length"])
            train_loader = DataLoader(train_dataset, batch_size=self.ner_cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)

            optimizer = get_optimizer(model, self.config)
            total_steps = len(train_loader) * 2
            scheduler = get_linear_schedule_with_warmup(optimizer, 50, total_steps)
            scaler = GradScaler(enabled=self.ner_cfg["fp16"] and torch.cuda.is_available())

            model.train()
            for _ in range(2):
                train_epoch(model, train_loader, optimizer, scheduler, scaler, device, self.ner_cfg["fp16"])

            val_f1_after, _ = evaluate_model(model, val_loader, id2label, label_list, device)
            logger.info(f"Val F1 after iteration {iteration}: {val_f1_after:.4f}")

            iteration_logs.append({
                "iteration": iteration,
                "examples_added": len(accepted),
                "val_f1_before": round(val_f1_before, 4),
                "val_f1_after": round(val_f1_after, 4),
            })

        # Save semi-supervised model
        model.save(os.path.join(output_dir, "ner_bert_base_semisup.pt"))

        final_f1, _ = evaluate_model(model, val_loader, id2label, label_list, device)
        expansion_log = {
            "iterations": iteration_logs,
            "total_examples_added": total_added,
            "final_val_f1": round(final_f1, 4),
        }

        log_path = os.path.join(self.paths["results"], "semisup_expansion_log.json")
        os.makedirs(self.paths["results"], exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(expansion_log, f, indent=2)

        logger.info(f"Semi-supervised expansion complete. Total added: {total_added}, Final F1: {final_f1:.4f}")
        return model
