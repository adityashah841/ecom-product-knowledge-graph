# E-Commerce Product Knowledge Graph

## Overview

This project builds a complete, end-to-end product knowledge graph pipeline over 75,000 Amazon e-commerce listings. Starting from raw product titles, it applies BERT-based named entity recognition to extract structured entities (brands, categories, colors, materials, attributes), uses a bi-encoder model to deduplicate product variants, and constructs a queryable knowledge graph with over 50,000 nodes and 6 relationship types.

The pipeline demonstrates the full ML lifecycle: rule-based silver label bootstrapping, supervised fine-tuning, semi-supervised pseudo-labeling for training data expansion, knowledge graph construction in NetworkX with optional Neo4j ingestion, and model compression via BERT-base → DistilBERT knowledge distillation. All training is designed for RTX 3050 (4GB VRAM) using fp16 and gradient checkpointing.

## Architecture

```
Raw Products (Amazon ESCI / McAuley-Lab Amazon Reviews 2023)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Pipeline                                     │
│  • Download 75K product listings (balanced across 20 cats) │
│  • Rule-based silver labeling (brand field + color/mat dict)│
│  • Train/val/test split → NER JSONL files                   │
│  • Build positive + hard-negative matching pairs            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 2: NER Fine-Tuning (BERT-base)                       │
│  • BertForTokenClassification, 6 entity types (BIO tags)   │
│  • fp16 + gradient checkpointing + 8-bit AdamW             │
│  • seqeval evaluation → results/ner_results.json            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 3: Semi-Supervised Expansion (3 iterations)          │
│  • Pseudo-label unlabeled pool at confidence ≥ 0.90        │
│  • Retrain 2 epochs per iteration                           │
│  • Logs expansion stats → results/semisup_expansion_log.json│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 4: Bi-Encoder Deduplication (Sentence-BERT)          │
│  • Fine-tune all-MiniLM-L6-v2 with MNRL + BCE              │
│  • FAISS flat index for efficient nearest-neighbor search   │
│  • Threshold sweep 0.70–0.95 → results/matching_results.json│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 5: Knowledge Distillation (BERT-base → DistilBERT)   │
│  • Soft KL divergence loss (T=4) + hard CE loss (α=0.7)    │
│  • ~40% smaller, ~60% faster, <5% F1 drop                  │
│  • results/distillation_results.json                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 6: Knowledge Graph Construction                      │
│  • Batch NER inference on all 75K products                  │
│  • VARIANT_OF edges from bi-encoder duplicate pairs         │
│  • NetworkX DiGraph → results/product_kg.graphml            │
│  • pyvis interactive HTML + matplotlib overview chart       │
│  • Optional Neo4j ingestion                                 │
└─────────────────────────────────────────────────────────────┘
```

## Key Results

| Component | Metric | Value |
|---|---|---|
| NER (BERT-base, supervised) | Overall F1 | **1.0000** |
| NER (BERT-base, semi-supervised) | Overall F1 | **1.0000** |
| Bi-Encoder Matching | F1 @ 0.85 | **1.0000** |
| Bi-Encoder Matching | F1 @ 0.90 | 0.9997 |
| Bi-Encoder Matching | F1 @ 0.95 | 0.9873 |
| Distillation (DistilBERT student) | Size | 265 MB vs 436 MB teacher |
| Distillation | Speedup | **1.73x** faster inference |
| Distillation | Compression | **1.64x** smaller |
| Knowledge Graph | Nodes | **50,458** |
| Knowledge Graph | Edges | **442,981** |

## Knowledge Graph Schema

**Node types:**
- `Product` — individual product listing (asin, title, confidence_score)
- `Brand` — manufacturer / seller brand (name)
- `Category` — product category / type (name)
- `Attribute` — product attribute (name)
- `Color` — color entity (name)
- `Material` — material entity (name)

**Edge types:**
- `(Product) -[MADE_BY]-> (Brand)`
- `(Product) -[BELONGS_TO]-> (Category)`
- `(Product) -[HAS_ATTRIBUTE]-> (Attribute)`
- `(Product) -[HAS_COLOR]-> (Color)`
- `(Product) -[MADE_FROM]-> (Material)`
- `(Product) -[VARIANT_OF]-> (Product)` — from bi-encoder deduplication

**Example subgraph:**
```
[Nike Air Max 270] -MADE_BY->    [Nike]
                   -BELONGS_TO-> [Running Shoes]
                   -HAS_COLOR->  [Black]
                   -MADE_FROM->  [Mesh]
                   -VARIANT_OF-> [Nike Air Max 270 (White)]
```

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ecom-product-knowledge-graph
cd ecom-product-knowledge-graph
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA 11.8+ (optional, CPU fallback available), 8GB RAM minimum.

## Usage

**Run the full pipeline:**
```bash
bash scripts/run_pipeline.sh
```

**Resume from a specific stage:**
```bash
python -m src.pipeline --config config/config.yaml --skip_to matching
```

**Dry run (see what would execute):**
```bash
python -m src.pipeline --config config/config.yaml --dry_run
```

**Download data only:**
```bash
bash scripts/download_data.sh
```

**Run individual components:**
```bash
# NER training only
python -m src.ner.train --config config/config.yaml --output_dir models/ --device cuda

# NER inference on a product title
python -c "
from src.ner.predict import predict_entities
# (load model and tokenizer first)
"
```

## Dataset

The pipeline uses the **Amazon ESCI** dataset (first tried) or falls back to **McAuley-Lab Amazon Reviews 2023 — Clothing, Shoes & Jewelry**. Both provide product titles, brand fields, and category metadata.

**Sampling strategy:** 75,000 products sampled proportionally across the top-20 product categories to ensure category balance.

**Silver label generation (rule-based bootstrapping):**
- Brand field → `B-BRAND / I-BRAND` tags by string matching in title
- Category field → `B-CATEGORY / I-CATEGORY` tags
- 200-entry color dictionary → `B-COLOR` tags
- 150-entry material dictionary → `B-MATERIAL` tags
- All other tokens → `O`

This produces noisy but structured training data without manual annotation, following the distant supervision paradigm. The semi-supervised stage then refines the model iteratively using high-confidence pseudo-labels.

## Technical Details

**VRAM management (RTX 3050, 4GB):**
- `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`
- fp16 training via `torch.cuda.amp.GradScaler`
- Gradient checkpointing on BERT encoder
- 8-bit AdamW via `bitsandbytes` (when available)
- Batch sizes: NER=16, Distillation=8 (teacher+student), Matching=32, Inference=64

**Distillation setup:**
- Teacher: `bert-base-uncased` fine-tuned on NER (frozen)
- Student: `distilbert-base-uncased` (6 layers vs 12)
- Loss: `α=0.7 * KL(student/T ‖ teacher/T) + 0.3 * CE(student, hard_labels)` where T=4.0

**Semi-supervised loop:**
- Confidence threshold: 0.90 (minimum over all non-O predicted tokens per example)
- 3 iterations, 2 retraining epochs per iteration
- Unlabeled pool: products not in any labeled split

## Repository Structure

```
ecom-product-knowledge-graph/
├── config/config.yaml          # All hyperparameters
├── data/samples/               # 500-row sample CSV (committed)
├── src/
│   ├── data/                   # Downloader + preprocessor
│   ├── ner/                    # BERT NER: dataset, model, train, eval, predict
│   ├── matching/               # Bi-encoder: dataset, model, train, eval, predict
│   ├── semi_supervised/        # Pseudo-labeling loop
│   ├── distillation/           # Teacher-student distillation + eval
│   ├── graph/                  # KG builder, Neo4j loader, visualizer
│   └── pipeline.py             # End-to-end orchestration
├── notebooks/                  # Exploration and analysis notebooks
├── results/                    # JSON results + graph files + visualizations
└── scripts/                    # Shell scripts for pipeline execution
```

## Results

![Graph Overview](results/graph_overview.png)

See [`results/graph_visualization.html`](results/graph_visualization.html) for the interactive knowledge graph visualization.
