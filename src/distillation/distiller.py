"""
BERT-base teacher → DistilBERT student distillation for NER.
Loss = alpha * KL_divergence(student/T, teacher/T) + (1-alpha) * CrossEntropy(student, hard_labels)
"""

import os
import logging
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from transformers import DistilBertForTokenClassification, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.ner.dataset import NERDataset, collate_fn
from src.ner.model import ProductNERModel

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class DistilledNERStudent(nn.Module):
    def __init__(self, model_name: str, num_labels: int, gradient_checkpointing: bool = False):
        super().__init__()
        self.model = DistilBertForTokenClassification.from_pretrained(model_name, num_labels=num_labels)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, labels=None):
        # DistilBERT doesn't use token_type_ids
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))
        return self


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Combined distillation + hard-label loss.
    KL div operates on valid (non-masked) positions only.
    """
    T = temperature
    valid_mask = hard_labels != -100

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    # Hard label loss
    loss_ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        hard_labels.view(-1),
        ignore_index=-100,
    )

    # Soft target KL loss on valid tokens
    s_log = F.log_softmax(student_logits[valid_mask] / T, dim=-1)
    t_soft = F.softmax(teacher_logits[valid_mask] / T, dim=-1)
    loss_kl = F.kl_div(s_log, t_soft, reduction="batchmean") * (T ** 2)

    return alpha * loss_kl + (1 - alpha) * loss_ce


def distill(config_path: str, output_dir: str, device_str: str):
    config = load_config(config_path)
    d_cfg = config["distillation"]
    ner_cfg = config["ner"]
    paths = config["paths"]

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available() and device_str == "cuda":
        logger.warning("CUDA not available — falling back to CPU")

    label_list = ner_cfg["label_list"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(ner_cfg["model_name"])

    # Load teacher (frozen)
    teacher_path = os.path.join(output_dir, "ner_bert_base_semisup.pt")
    if not os.path.exists(teacher_path):
        teacher_path = os.path.join(output_dir, "ner_bert_base_best.pt")
    teacher = ProductNERModel(ner_cfg["model_name"], len(label_list)).to(device)
    teacher.load(teacher_path, device)
    teacher.eval()
    teacher.requires_grad_(False)
    logger.info(f"Loaded teacher from {teacher_path}")

    # Initialize student
    student = DistilledNERStudent(
        d_cfg["student_model"],
        num_labels=len(label_list),
        gradient_checkpointing=d_cfg["gradient_checkpointing"],
    ).to(device)

    train_dataset = NERDataset(
        os.path.join(paths["processed_data"], "ner_train.json"),
        tokenizer, label2id, config["data"]["max_seq_length"],
    )
    val_dataset = NERDataset(
        os.path.join(paths["processed_data"], "ner_val.json"),
        tokenizer, label2id, config["data"]["max_seq_length"],
    )

    train_loader = DataLoader(train_dataset, batch_size=d_cfg["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=d_cfg["batch_size"], shuffle=False, collate_fn=collate_fn, num_workers=0)

    optimizer = AdamW(student.parameters(), lr=float(d_cfg["learning_rate"]))
    total_steps = len(train_loader) * d_cfg["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, 100, total_steps)
    scaler = GradScaler(enabled=d_cfg["fp16"] and torch.cuda.is_available())

    os.makedirs(output_dir, exist_ok=True)
    best_val_loss = float("inf")
    torch.backends.cudnn.benchmark = True

    for epoch in range(d_cfg["num_epochs"]):
        student.train()
        torch.cuda.empty_cache()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Distill epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.no_grad():
                with autocast():
                    teacher_out = teacher(
                        batch["input_ids"], batch["attention_mask"], batch["token_type_ids"]
                    )
                teacher_logits = teacher_out.logits.detach()

            if d_cfg["fp16"] and torch.cuda.is_available():
                with autocast():
                    student_out = student(batch["input_ids"], batch["attention_mask"])
                    loss = distillation_loss(
                        student_out.logits, teacher_logits, batch["labels"],
                        d_cfg["temperature"], d_cfg["alpha"],
                    )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                student_out = student(batch["input_ids"], batch["attention_mask"])
                loss = distillation_loss(
                    student_out.logits, teacher_logits, batch["labels"],
                    d_cfg["temperature"], d_cfg["alpha"],
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)

        # Validation
        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast():
                    student_out = student(batch["input_ids"], batch["attention_mask"], labels=batch["labels"])
                val_loss += student_out.loss.item()

        avg_val = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            student.save(os.path.join(output_dir, "ner_distilbert_student.pt"))
            logger.info("  Saved best student checkpoint")

    logger.info("Distillation training complete")
    return student, teacher, tokenizer, label_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--output_dir", default="models/")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    distill(args.config, args.output_dir, args.device)
