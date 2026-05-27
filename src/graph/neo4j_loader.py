"""Optional Neo4j ingestion. Skips gracefully if Neo4j is not running."""

import logging
from itertools import islice

import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch


def load_to_neo4j(graph: nx.DiGraph, uri: str, user: str, password: str, batch_size: int = 1000):
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j driver not installed — skipping Neo4j load")
        return

    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
    except Exception as e:
        logger.warning(f"Neo4j not available ({e}) — graph saved to GraphML only.")
        return

    logger.info("Connected to Neo4j. Loading graph...")

    with driver.session() as session:
        # Create uniqueness constraints
        node_types = set(d.get("type", "Unknown") for _, d in graph.nodes(data=True))
        for nt in node_types:
            constraint_prop = "asin" if nt == "Product" else "name"
            try:
                session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{nt}) REQUIRE n.{constraint_prop} IS UNIQUE")
            except Exception:
                pass

        # Insert nodes in batches
        nodes = list(graph.nodes(data=True))
        for batch in _batched(nodes, batch_size):
            for node_id, data in batch:
                node_type = data.get("type", "Unknown")
                props = {k: v for k, v in data.items() if k != "type"}
                props["_id"] = str(node_id)
                session.run(
                    f"MERGE (n:{node_type} {{_id: $id}}) SET n += $props",
                    id=str(node_id), props=props,
                )

        # Insert edges in batches
        edges = list(graph.edges(data=True))
        for batch in _batched(edges, batch_size):
            for src, dst, data in batch:
                relation = data.get("relation", "RELATED_TO").replace(" ", "_")
                session.run(
                    f"MATCH (a {{_id: $src}}), (b {{_id: $dst}}) MERGE (a)-[:{relation}]->(b)",
                    src=str(src), dst=str(dst),
                )

    driver.close()
    logger.info(f"Loaded {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges to Neo4j")
