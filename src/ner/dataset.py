"""PyTorch Dataset for NER token classification."""

import json
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast


class NERDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: BertTokenizerFast, label2id: dict, max_length: int = 128):
        self.samples = []
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample["tokens"]
        word_labels = sample["labels"]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev_word_id:
                tag = word_labels[word_id] if word_id < len(word_labels) else "O"
                label_ids.append(self.label2id.get(tag, 0))
            else:
                label_ids.append(-100)
            prev_word_id = word_id

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding.get("token_type_ids", torch.zeros(self.max_length, dtype=torch.long)).squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def collate_fn(batch):
    return {key: torch.stack([item[key] for item in batch]) for key in batch[0]}
