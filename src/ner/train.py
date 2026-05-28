"""
NER fine-tuning script for BERT-base token classification.
Supports fp16, gradient checkpointing, 8-bit optimizer, and checkpoint resumption.
"""

import os
import json
import logging
import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast

from src.ner.dataset import NERDataset, collate_fn
from src.ner.model import ProductNERModel
from src.ner.evaluate import evaluate_model

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_optimizer(model, config: dict):
    lr = float(config["ner"]["learning_rate"])
    wd = float(config["ner"]["weight_decay"])
    from torch.optim import AdamW
    return AdamW(model.parameters(), lr=lr, weight_decay=wd)


def train_epoch(model, loader, optimizer, scheduler, scaler, device, fp16: bool):
    model.train()
    torch.cuda.empty_cache()
    total_loss = 0.0

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        if fp16:
            with autocast():
                outputs = model(**batch)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train(config_path: str, output_dir: str, device_str: str, resume: bool = False):
    config = load_config(config_path)
    ner_cfg = config["ner"]
    paths = config["paths"]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and device_str == "cuda":
        logger.warning("CUDA not available — falling back to CPU")

    label_list = ner_cfg["label_list"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    tokenizer = BertTokenizerFast.from_pretrained(ner_cfg["model_name"])

    train_dataset = NERDataset(
        os.path.join(paths["processed_data"], "ner_train.json"),
        tokenizer, label2id, config["data"]["max_seq_length"],
    )
    val_dataset = NERDataset(
        os.path.join(paths["processed_data"], "ner_val.json"),
        tokenizer, label2id, config["data"]["max_seq_length"],
    )

    batch_size = ner_cfg["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = ProductNERModel(
        ner_cfg["model_name"],
        num_labels=len(label_list),
        gradient_checkpointing=ner_cfg["gradient_checkpointing"],
    ).to(device)

    start_epoch = 0
    best_val_f1 = 0.0

    resume_path = os.path.join(output_dir, "ner_bert_base_final.pt")
    if resume and os.path.exists(resume_path):
        model.load(resume_path, device)
        logger.info(f"Resumed from {resume_path}")

    optimizer = get_optimizer(model, config)
    total_steps = len(train_loader) * ner_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, ner_cfg["warmup_steps"], total_steps)
    scaler = GradScaler(enabled=ner_cfg["fp16"] and torch.cuda.is_available())

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(paths["results"], exist_ok=True)
    log_path = os.path.join(paths["results"], "ner_training_log.jsonl")

    torch.backends.cudnn.benchmark = True

    for epoch in range(start_epoch, ner_cfg["num_epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, ner_cfg["fp16"])
        val_f1, val_loss = evaluate_model(model, val_loader, id2label, label_list, device)

        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_f1": round(val_f1, 4),
        }
        logger.info(f"Epoch {epoch+1}: {log_entry}")

        with open(log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save(os.path.join(output_dir, "ner_bert_base_best.pt"))
            logger.info(f"  New best model saved (F1={val_f1:.4f})")

    model.save(os.path.join(output_dir, "ner_bert_base_final.pt"))
    logger.info(f"Training complete. Best val F1: {best_val_f1:.4f}")
    return model, tokenizer, label_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output_dir", default="models/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train(args.config, args.output_dir, args.device, args.resume)
