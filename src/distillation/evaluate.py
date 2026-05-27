"""Compare teacher (BERT-base) vs student (DistilBERT) on NER test set."""

import json
import logging
import os
import time

import torch
import yaml
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from seqeval.metrics import f1_score

from src.ner.dataset import NERDataset, collate_fn
from src.ner.evaluate import decode_predictions
from src.ner.model import ProductNERModel
from src.distillation.distiller import DistilledNERStudent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_model_size_mb(model: torch.nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return round(total / 1e6, 2)


def measure_inference(model, data_loader: DataLoader, device: torch.device, is_student: bool = False, n_batches: int = 100) -> tuple[float, float]:
    """Returns (f1, ms_per_batch)."""
    model.eval()
    all_preds, all_true = [], []
    times = []

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            start = time.perf_counter()
            with autocast():
                if is_student:
                    outputs = model(batch["input_ids"], batch["attention_mask"])
                else:
                    outputs = model(batch["input_ids"], batch["attention_mask"], batch["token_type_ids"])
            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)
            preds, trues = decode_predictions(outputs.logits, batch["labels"], {})
            all_preds.extend(preds)
            all_true.extend(trues)
            if i >= n_batches:
                break

    # id2label not needed — decode_predictions will fall back to "O" for unknown ids
    # Use seqeval on string labels directly
    f1 = f1_score(all_true, all_preds) if all_true else 0.0
    avg_ms = round(sum(times) / len(times), 2) if times else 0.0
    return round(f1, 4), avg_ms


def run_comparison(config_path: str, models_dir: str, results_path: str, device_str: str = "cuda"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ner_cfg = config["ner"]
    paths = config["paths"]
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    label_list = ner_cfg["label_list"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(ner_cfg["model_name"])

    test_dataset = NERDataset(
        os.path.join(paths["processed_data"], "ner_test.json"),
        tokenizer, label2id, config["data"]["max_seq_length"],
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # Teacher
    teacher_path = os.path.join(models_dir, "ner_bert_base_semisup.pt")
    if not os.path.exists(teacher_path):
        teacher_path = os.path.join(models_dir, "ner_bert_base_best.pt")
    teacher = ProductNERModel(ner_cfg["model_name"], len(label_list)).to(device)
    teacher.load(teacher_path, device)
    teacher_f1, teacher_ms = measure_inference(teacher, test_loader, device, is_student=False)
    teacher_size = get_model_size_mb(teacher)

    # Student
    student_path = os.path.join(models_dir, "ner_distilbert_student.pt")
    student = DistilledNERStudent(config["distillation"]["student_model"], len(label_list)).to(device)
    student.load(student_path, device)
    student_f1, student_ms = measure_inference(student, test_loader, device, is_student=True)
    student_size = get_model_size_mb(student)

    compression_ratio = round(teacher_size / max(student_size, 0.01), 2)
    speedup = round(teacher_ms / max(student_ms, 0.01), 2)
    f1_retention = round(student_f1 / max(teacher_f1, 0.001) * 100, 1)

    results = {
        "teacher": {
            "model": ner_cfg["model_name"],
            "size_mb": teacher_size,
            "f1": teacher_f1,
            "inference_ms_per_batch": teacher_ms,
        },
        "student": {
            "model": config["distillation"]["student_model"],
            "size_mb": student_size,
            "f1": student_f1,
            "inference_ms_per_batch": student_ms,
        },
        "compression_ratio": compression_ratio,
        "speedup": speedup,
        "f1_retention_pct": f1_retention,
    }

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Distillation results saved to {results_path}")
    logger.info(f"Teacher F1: {teacher_f1} | Student F1: {student_f1} | Speedup: {speedup}x | Compression: {compression_ratio}x")
    return results
