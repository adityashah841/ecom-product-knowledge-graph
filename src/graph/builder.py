"""
Knowledge Graph construction using NER predictions and bi-encoder deduplication.
Graph schema:
  Nodes: Product, Brand, Category, Attribute, Color, Material
  Edges: MADE_BY, BELONGS_TO, HAS_ATTRIBUTE, HAS_COLOR, MADE_FROM, VARIANT_OF
"""

import json
import logging
import os
from collections import defaultdict
from typing import Optional

import networkx as nx
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


ENTITY_EDGE_MAP = {
    "BRAND": "MADE_BY",
    "CATEGORY": "BELONGS_TO",
    "ATTRIBUTE": "HAS_ATTRIBUTE",
    "COLOR": "HAS_COLOR",
    "MATERIAL": "MADE_FROM",
}


class KnowledgeGraphBuilder:
    def __init__(self, config: dict):
        self.config = config
        self.g_cfg = config["graph"]
        self.G = nx.DiGraph()
        self._node_cache: dict[str, str] = {}  # canonical_name -> node_id

    def _node_id(self, name: str, node_type: str) -> str:
        key = f"{node_type}::{name.lower().strip()}"
        if key not in self._node_cache:
            node_id = f"{node_type}_{len(self._node_cache)}"
            self._node_cache[key] = node_id
        return self._node_cache[key]

    def add_product(self, asin: str, title: str, confidence: float = 1.0):
        node_id = f"PRODUCT_{asin}"
        self.G.add_node(node_id, type="Product", title=title, asin=asin, confidence=confidence)
        return node_id

    def add_entity_node(self, name: str, entity_type: str, confidence: float) -> Optional[str]:
        if confidence < self.g_cfg["min_entity_confidence"]:
            return None
        name = name.lower().strip()
        if not name or name in ("nan", "none", "unknown"):
            return None
        node_id = self._node_id(name, entity_type)
        if not self.G.has_node(node_id):
            self.G.add_node(node_id, type=entity_type, name=name)
        return node_id

    def add_relation(self, product_node_id: str, entity_node_id: str, relation: str, confidence: float = 1.0):
        self.G.add_edge(product_node_id, entity_node_id, relation=relation, confidence=round(confidence, 4))

    def build_from_predictions(self, predictions: list[dict]):
        """
        predictions: list of {asin, title, entities: [{entity, type, confidence}]}
        """
        for record in predictions:
            asin = record.get("asin", f"ASIN_{len(self.G)}")
            product_id = self.add_product(asin, record["title"])

            for ent in record.get("entities", []):
                entity_type = ent.get("type", "")
                relation = ENTITY_EDGE_MAP.get(entity_type)
                if relation is None:
                    continue
                ent_node_id = self.add_entity_node(ent["entity"], entity_type, ent["confidence"])
                if ent_node_id:
                    self.add_relation(product_id, ent_node_id, relation, ent["confidence"])

    def add_variant_edges(self, duplicate_pairs: list[tuple[int, int, float]], product_ids: list[str]):
        """Add VARIANT_OF edges from bi-encoder duplicate pairs."""
        for idx_a, idx_b, sim in duplicate_pairs:
            if idx_a < len(product_ids) and idx_b < len(product_ids):
                self.G.add_edge(
                    product_ids[idx_a], product_ids[idx_b],
                    relation="VARIANT_OF", confidence=round(sim, 4),
                )

    def compute_stats(self) -> dict:
        node_types = defaultdict(int)
        edge_types = defaultdict(int)

        for _, data in self.G.nodes(data=True):
            node_types[data.get("type", "Unknown")] += 1
        for _, _, data in self.G.edges(data=True):
            edge_types[data.get("relation", "Unknown")] += 1

        degrees = [d for _, d in self.G.degree()]
        avg_degree = round(sum(degrees) / max(len(degrees), 1), 2)
        num_components = nx.number_weakly_connected_components(self.G)

        brand_nodes = [(n, d) for n, d in self.G.nodes(data=True) if d.get("type") == "Brand"]
        top_brands = sorted(
            [(data.get("name", n), self.G.degree(n)) for n, data in brand_nodes],
            key=lambda x: -x[1],
        )[:10]

        stats = {
            "total_nodes": self.G.number_of_nodes(),
            "total_edges": self.G.number_of_edges(),
            "nodes_by_type": dict(node_types),
            "edges_by_type": dict(edge_types),
            "average_degree": avg_degree,
            "num_weakly_connected_components": num_components,
            "top_10_brands_by_degree": top_brands,
        }

        for key, val in stats.items():
            if key != "top_10_brands_by_degree":
                logger.info(f"  {key}: {val}")

        return stats

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        nx.write_graphml(self.G, path)
        logger.info(f"Graph saved to {path} ({self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges)")


def build_graph_from_data(config_path: str, ner_model, ner_tokenizer, id2label: dict, bi_model, bi_tokenizer, device) -> dict:
    """
    Full graph construction pipeline:
    1. Batch NER inference over all products
    2. Run bi-encoder deduplication
    3. Build and save the graph
    """
    import torch
    from src.ner.predict import batch_predict
    from src.matching.predict import find_duplicates

    config = load_config(config_path)
    paths = config["paths"]
    g_cfg = config["graph"]

    raw_parquet = os.path.join(paths["raw_data"], "products_sampled.parquet")
    if not os.path.exists(raw_parquet):
        raise FileNotFoundError(f"Raw parquet not found at {raw_parquet}")

    df = pd.read_parquet(raw_parquet)
    title_col = "title" if "title" in df.columns else "product_title"
    asin_col = "asin" if "asin" in df.columns else df.columns[0]

    df = df.dropna(subset=[title_col])
    df = df.head(g_cfg["max_nodes"])
    titles = df[title_col].astype(str).tolist()
    asins = df[asin_col].astype(str).tolist() if asin_col in df.columns else [str(i) for i in range(len(df))]

    logger.info(f"Running NER inference on {len(titles)} products...")
    predictions = []
    batch_size = 64
    ner_model.eval()
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i:i + batch_size]
        batch_asins = asins[i:i + batch_size]
        batch_preds = batch_predict(batch_titles, ner_model, ner_tokenizer, id2label, device)
        for j, (asin, title, entities) in enumerate(zip(batch_asins, batch_titles, batch_preds)):
            predictions.append({"asin": asin, "title": title, "entities": entities})

    builder = KnowledgeGraphBuilder(config)
    builder.build_from_predictions(predictions)

    # Deduplication
    logger.info("Running bi-encoder deduplication...")
    product_ids = [f"PRODUCT_{p['asin']}" for p in predictions]
    threshold = config["matching"]["similarity_threshold"]
    try:
        duplicate_pairs = find_duplicates(titles, bi_model, bi_tokenizer, device, threshold)
        builder.add_variant_edges(duplicate_pairs, product_ids)
    except Exception as e:
        logger.warning(f"Deduplication failed: {e} — skipping VARIANT_OF edges")

    stats = builder.compute_stats()
    graph_path = os.path.join(paths["results"], "product_kg.graphml")
    builder.save(graph_path)

    stats_path = os.path.join(paths["results"], "graph_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Graph stats saved to {stats_path}")

    return stats
