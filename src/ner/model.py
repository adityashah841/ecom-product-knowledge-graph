"""BERT-base for token classification (NER)."""

import torch
import torch.nn as nn
from transformers import BertForTokenClassification, BertConfig


class ProductNERModel(nn.Module):
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1, gradient_checkpointing: bool = False):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        if gradient_checkpointing:
            self.bert.bert.gradient_checkpointing_enable()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: torch.device):
        self.load_state_dict(torch.load(path, map_location=device))
        return self
