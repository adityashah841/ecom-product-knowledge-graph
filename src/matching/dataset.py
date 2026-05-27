"""PyTorch Dataset for bi-encoder text matching."""

import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class MatchingDataset(Dataset):
    def __init__(self, file_path: str, tokenizer_name: str = "sentence-transformers/all-MiniLM-L6-v2", max_length: int = 128):
        self.samples = []
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        enc_a = self.tokenizer(
            sample["text_a"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        enc_b = self.tokenizer(
            sample["text_b"], truncation=True, max_length=self.max_length,
            padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float),
        }


def matching_collate_fn(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
