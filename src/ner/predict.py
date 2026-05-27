"""NER inference: extract entities from raw product text."""

import torch
import torch.nn.functional as F
from transformers import BertTokenizerFast


def predict_entities(text: str, model, tokenizer: BertTokenizerFast, id2label: dict, device: torch.device) -> list[dict]:
    """
    Extract named entities from a product title string.
    Returns list of {"entity": str, "type": str, "start": int, "end": int, "confidence": float}.
    Aggregates subword predictions back to word level using the first subword's label.
    """
    model.eval()
    words = text.split()

    encoding = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)

    logits = outputs.logits[0]
    probs = F.softmax(logits, dim=-1)
    word_ids = encoding["input_ids"].size(1)

    # Map predictions back to word level
    word_preds = {}
    prev_word_id = None
    for token_idx, word_id in enumerate(encoding["input_ids"][0].tolist()):
        wid = tokenizer.convert_ids_to_tokens([encoding["input_ids"][0][token_idx]])[0]
        # Use word_ids from the encoding
    encoding_cpu = tokenizer(words, is_split_into_words=True, truncation=True, max_length=128)
    token_word_ids = encoding_cpu.word_ids()

    word_preds = {}
    for token_idx, word_id in enumerate(token_word_ids):
        if word_id is None:
            continue
        if word_id not in word_preds:
            label_id = torch.argmax(probs[token_idx]).item()
            confidence = probs[token_idx][label_id].item()
            word_preds[word_id] = (id2label.get(label_id, "O"), confidence)

    # Build entity spans from BIO sequence
    entities = []
    current_entity = None

    for word_idx, word in enumerate(words):
        if word_idx not in word_preds:
            tag, conf = "O", 1.0
        else:
            tag, conf = word_preds[word_idx]

        if tag.startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "entity": word,
                "type": tag[2:],
                "start": word_idx,
                "end": word_idx + 1,
                "confidence": conf,
                "_tokens": [word],
                "_confs": [conf],
            }
        elif tag.startswith("I-") and current_entity and tag[2:] == current_entity["type"]:
            current_entity["_tokens"].append(word)
            current_entity["_confs"].append(conf)
            current_entity["end"] = word_idx + 1
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None

    if current_entity:
        entities.append(current_entity)

    # Clean up internal fields and compute mean confidence
    result = []
    for ent in entities:
        result.append({
            "entity": " ".join(ent["_tokens"]),
            "type": ent["type"],
            "start": ent["start"],
            "end": ent["end"],
            "confidence": round(sum(ent["_confs"]) / len(ent["_confs"]), 4),
        })

    return result


def batch_predict(texts: list[str], model, tokenizer, id2label: dict, device: torch.device) -> list[list[dict]]:
    """Predict entities for a batch of texts."""
    return [predict_entities(text, model, tokenizer, id2label, device) for text in texts]
