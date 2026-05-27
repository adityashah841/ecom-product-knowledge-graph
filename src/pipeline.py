"""
End-to-end pipeline orchestration.
Each stage writes a .done marker so the pipeline can be resumed.
Usage: python -m src.pipeline --config config/config.yaml [--skip_to STAGE] [--dry_run]
"""

import argparse
import logging
import os
import sys

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGES = ["data", "ner", "semisup", "matching", "distillation", "graph"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def marker_path(results_dir: str, stage: str) -> str:
    return os.path.join(results_dir, f".{stage}.done")


def is_done(results_dir: str, stage: str) -> bool:
    return os.path.exists(marker_path(results_dir, stage))


def mark_done(results_dir: str, stage: str):
    os.makedirs(results_dir, exist_ok=True)
    with open(marker_path(results_dir, stage), "w") as f:
        f.write("done")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory // 1e6:.0f} MB VRAM)")
        return torch.device("cuda")
    logger.warning("CUDA not available — running on CPU (will be slow)")
    return torch.device("cpu")


def run_pipeline(config_path: str, skip_to: str = None, dry_run: bool = False):
    config = load_config(config_path)
    results_dir = config["paths"]["results"]
    models_dir = config["paths"]["models"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    device = get_device()
    started = skip_to is None

    label_list = config["ner"]["label_list"]
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    ner_model = ner_tokenizer = bi_model = bi_tokenizer = None

    for stage in STAGES:
        if not started:
            if stage == skip_to:
                started = True
            else:
                logger.info(f"[SKIP] {stage}")
                continue

        if is_done(results_dir, stage) and not dry_run:
            logger.info(f"[DONE] {stage} (cached)")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"STAGE: {stage.upper()}")
        logger.info(f"{'='*50}")

        if dry_run:
            logger.info(f"[DRY RUN] Would execute stage: {stage}")
            continue

        try:
            if stage == "data":
                from src.data.downloader import main as download_main
                download_main(config_path)
                from src.data.preprocessor import build_matching_pairs
                import pandas as pd
                parquet_path = os.path.join(config["paths"]["raw_data"], "products_sampled.parquet")
                df = pd.read_parquet(parquet_path)
                pairs_path = os.path.join(config["paths"]["processed_data"], "matching_pairs.json")
                build_matching_pairs(df, pairs_path, config["data"]["seed"])
                mark_done(results_dir, stage)

            elif stage == "ner":
                from src.ner.train import train
                from src.ner.dataset import NERDataset, collate_fn
                from src.ner.evaluate import evaluate_and_save
                from torch.utils.data import DataLoader
                from transformers import BertTokenizerFast

                ner_model, ner_tokenizer, label_list_out = train(config_path, models_dir, str(device))

                ner_tokenizer_loaded = BertTokenizerFast.from_pretrained(config["ner"]["model_name"])
                test_ds = NERDataset(
                    os.path.join(config["paths"]["processed_data"], "ner_test.json"),
                    ner_tokenizer_loaded, label2id, config["data"]["max_seq_length"],
                )
                test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=0)
                evaluate_and_save(ner_model, test_loader, id2label, label_list, device,
                                  os.path.join(results_dir, "ner_results.json"))
                mark_done(results_dir, stage)

            elif stage == "semisup":
                from src.semi_supervised.pseudo_labeler import PseudoLabelingLoop
                from src.ner.model import ProductNERModel
                from transformers import BertTokenizerFast

                if ner_model is None:
                    ner_model = ProductNERModel(config["ner"]["model_name"], len(label_list)).to(device)
                    ner_model.load(os.path.join(models_dir, "ner_bert_base_best.pt"), device)
                    ner_tokenizer = BertTokenizerFast.from_pretrained(config["ner"]["model_name"])

                loop = PseudoLabelingLoop(config_path)
                ner_model = loop.run(ner_model, ner_tokenizer, id2label, label_list, device, models_dir)
                mark_done(results_dir, stage)

            elif stage == "matching":
                from src.matching.train import train as matching_train
                from src.matching.evaluate import run_evaluation
                from src.matching.model import BiEncoderModel
                from transformers import AutoTokenizer

                bi_model = matching_train(config_path, models_dir, str(device))
                run_evaluation(
                    config_path,
                    os.path.join(models_dir, "biencoder_best.pt"),
                    os.path.join(results_dir, "matching_results.json"),
                    str(device),
                )
                bi_tokenizer = AutoTokenizer.from_pretrained(config["matching"]["model_name"])
                mark_done(results_dir, stage)

            elif stage == "distillation":
                from src.distillation.distiller import distill
                from src.distillation.evaluate import run_comparison

                distill(config_path, models_dir, str(device))
                run_comparison(
                    config_path, models_dir,
                    os.path.join(results_dir, "distillation_results.json"),
                    str(device),
                )
                mark_done(results_dir, stage)

            elif stage == "graph":
                from src.ner.model import ProductNERModel
                from src.matching.model import BiEncoderModel
                from src.graph.builder import build_graph_from_data
                from src.graph.visualizer import plot_graph_overview, generate_interactive_html
                from src.graph.neo4j_loader import load_to_neo4j
                from transformers import BertTokenizerFast, AutoTokenizer
                import json
                import networkx as nx

                if ner_model is None:
                    semisup_path = os.path.join(models_dir, "ner_bert_base_semisup.pt")
                    best_path = os.path.join(models_dir, "ner_bert_base_best.pt")
                    ner_model = ProductNERModel(config["ner"]["model_name"], len(label_list)).to(device)
                    ner_model.load(semisup_path if os.path.exists(semisup_path) else best_path, device)
                    ner_tokenizer = BertTokenizerFast.from_pretrained(config["ner"]["model_name"])

                if bi_model is None:
                    bi_model = BiEncoderModel(config["matching"]["model_name"]).to(device)
                    bienc_path = os.path.join(models_dir, "biencoder_best.pt")
                    if os.path.exists(bienc_path):
                        bi_model.load(bienc_path, device)
                    bi_tokenizer = AutoTokenizer.from_pretrained(config["matching"]["model_name"])

                stats = build_graph_from_data(config_path, ner_model, ner_tokenizer, id2label, bi_model, bi_tokenizer, device)

                graphml_path = os.path.join(results_dir, "product_kg.graphml")
                G = nx.read_graphml(graphml_path)

                plot_graph_overview(stats, os.path.join(results_dir, "graph_overview.png"))
                generate_interactive_html(G, os.path.join(results_dir, "graph_visualization.html"),
                                          config["graph"]["visualization_sample"])

                # Try Neo4j (optional)
                load_to_neo4j(G, "bolt://localhost:7687", "neo4j", "password")

                mark_done(results_dir, stage)

        except torch.cuda.OutOfMemoryError as oom:
            logger.error(f"CUDA OOM in stage {stage}: {oom}")
            logger.error("Try reducing batch size in config/config.yaml")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Stage {stage} failed: {e}", exc_info=True)
            sys.exit(1)

    # Summary
    logger.info("\n" + "="*50)
    logger.info("PIPELINE COMPLETE — Output files:")
    for fname in [
        "ner_results.json", "matching_results.json",
        "distillation_results.json", "graph_stats.json",
        "product_kg.graphml", "graph_overview.png", "graph_visualization.html",
    ]:
        path = os.path.join(results_dir, fname)
        status = "OK" if os.path.exists(path) else "MISSING"
        logger.info(f"  [{status}] {path}")
    logger.info("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="E-Commerce Product Knowledge Graph Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--skip_to", default=None, choices=STAGES, help="Skip to a specific stage")
    parser.add_argument("--dry_run", action="store_true", help="Print what would run without executing")
    args = parser.parse_args()

    run_pipeline(args.config, args.skip_to, args.dry_run)
