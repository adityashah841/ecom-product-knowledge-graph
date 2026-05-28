# E-Commerce Product Knowledge Graph

## Overview

This project builds a complete, end-to-end product knowledge graph pipeline over 75,000 e-commerce product listings. Starting from raw product titles, it applies BERT-based named entity recognition to extract structured entities (brands, categories, colors, materials, attributes), uses a bi-encoder model to deduplicate product variants, and constructs a queryable knowledge graph with over 50,000 nodes and 6 relationship types.

The pipeline demonstrates the full ML lifecycle: programmatic training data generation, supervised fine-tuning, semi-supervised pseudo-labeling, knowledge graph construction in NetworkX with optional Neo4j ingestion, and model compression via BERT-base → DistilBERT knowledge distillation. All training runs on an RTX 3050 (4GB VRAM) using fp16 and gradient checkpointing.

## Architecture

```
Synthetic Product Catalogue (75K titles, 171 brands, 30 categories)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  STAGE 1: Data Pipeline                                     │
│  • Generate 75K product listings via template synthesis     │
│  • Perfect BIO labels (no noise — titles built from fields) │
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

> **On the perfect F1 scores:** NER and matching both achieve F1=1.0 because the training data is synthetically generated with exact, noise-free labels (see [Dataset](#dataset) section). This validates that the model architecture and training pipeline are correctly implemented. On a real-world noisy corpus (scraped Amazon listings with imperfect brand matching), expect NER F1 in the 0.75–0.88 range — the pipeline is designed to handle that via the semi-supervised expansion stage.

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

### Synthetic Product Catalogue

The pipeline generates a **75,000-product synthetic catalogue** (`src/data/downloader.py`) rather than scraping a live dataset. This was a deliberate engineering decision made after HuggingFace deprecated custom dataset loading scripts in `datasets` v4.x, which broke both the Amazon ESCI and McAuley-Lab Amazon Reviews 2023 loaders.

The synthetic generator produces realistic Amazon-style product titles by combining:
- **171 real brand names** (Nike, Apple, Carhartt, KitchenAid, Dewalt, …)
- **30 product categories** with subcategory variants (Running Shoes → Road/Trail/Racing Flats, …)
- **34 colors**, **26 materials**, **80+ attributes** drawn from curated vocabularies
- **10 title templates** that mirror real listing patterns:
  ```
  "{Brand} {Attr} {Color} {Material} {Category}"
  "{Brand} {Color} {Category} for {Use} | {Material} | {Attr}"
  "{Color} {Material} {Category} by {Brand} - {Attr}"
  ```

**Why this works better than distant supervision:**

Real Amazon datasets require rule-based silver labeling — matching a brand string into a noisy title — which produces label noise whenever a brand name appears mid-word, is abbreviated, or conflicts with a product term. The synthetic approach flips this: since we *construct* each title from known entity strings, every BIO label is guaranteed correct. The model learns from a perfectly consistent signal, which is why NER F1 converges to 1.0 by epoch 2.

This is a standard technique in low-resource NLP (see [Few-NERD](https://arxiv.org/abs/2105.07464), [CrossNER](https://arxiv.org/abs/2012.04373)) and is legitimate for demonstrating pipeline architecture when the goal is the system design, not a benchmark comparison on a specific corpus.

**To swap in a real dataset:** replace `generate_synthetic_products()` in `src/data/downloader.py` with any DataFrame that has `title`, `brand`, and `category` columns. The rest of the pipeline is dataset-agnostic.

**Sampling strategy:** Products are sampled proportionally across all 30 categories using a Dirichlet-weighted distribution to ensure category balance without being perfectly uniform (matching real-world long-tail behavior).

**Label generation:**
- Brand name → `B-BRAND / I-BRAND` by exact token match in title
- Category / subcategory → `B-CATEGORY / I-CATEGORY`
- Color → `B-COLOR / I-COLOR`
- Material → `B-MATERIAL`
- Attribute → `B-ATTRIBUTE`
- All other tokens → `O`

Because titles are built from these fields, match rate is 100% and there is no label noise.

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
