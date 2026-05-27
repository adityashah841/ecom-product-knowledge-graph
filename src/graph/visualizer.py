"""Graph visualization: matplotlib stats chart and pyvis interactive HTML."""

import logging
import os

import matplotlib.pyplot as plt
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NODE_COLORS = {
    "Product": "#4e79a7",
    "Brand": "#f28e2b",
    "Category": "#e15759",
    "Attribute": "#76b7b2",
    "Color": "#59a14f",
    "Material": "#edc948",
}


def plot_graph_overview(stats: dict, output_path: str):
    """Matplotlib bar chart of node/edge counts by type."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Product Knowledge Graph — Overview", fontsize=14, fontweight="bold")

    nodes_by_type = stats.get("nodes_by_type", {})
    edges_by_type = stats.get("edges_by_type", {})

    if nodes_by_type:
        types = list(nodes_by_type.keys())
        counts = list(nodes_by_type.values())
        colors = [NODE_COLORS.get(t, "#aaa") for t in types]
        axes[0].bar(types, counts, color=colors, edgecolor="white")
        axes[0].set_title(f"Nodes by Type (Total: {stats.get('total_nodes', 0):,})")
        axes[0].set_xlabel("Node Type")
        axes[0].set_ylabel("Count")
        axes[0].tick_params(axis="x", rotation=30)
        for i, v in enumerate(counts):
            axes[0].text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=8)

    if edges_by_type:
        rels = list(edges_by_type.keys())
        ecounts = list(edges_by_type.values())
        axes[1].barh(rels, ecounts, color="#6b6ecf", edgecolor="white")
        axes[1].set_title(f"Edges by Relation (Total: {stats.get('total_edges', 0):,})")
        axes[1].set_xlabel("Count")
        for i, v in enumerate(ecounts):
            axes[1].text(v + max(ecounts) * 0.01, i, f"{v:,}", va="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Graph overview saved to {output_path}")


def generate_interactive_html(graph: nx.DiGraph, output_path: str, n_sample: int = 500):
    """Generate pyvis interactive HTML visualization of a subgraph."""
    try:
        from pyvis.network import Network
    except ImportError:
        logger.warning("pyvis not available — skipping HTML visualization")
        return

    # Sample highest-degree nodes
    degrees = dict(graph.degree())
    top_nodes = sorted(degrees, key=lambda x: -degrees[x])[:n_sample]
    subgraph = graph.subgraph(top_nodes)

    net = Network(height="750px", width="100%", bgcolor="#1a1a2e", font_color="white", directed=True)
    net.set_options("""
    var options = {
      "nodes": {"borderWidth": 2, "shadow": true, "font": {"size": 12}},
      "edges": {"arrows": {"to": {"enabled": true, "scaleFactor": 0.8}}, "smooth": true},
      "physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 120}, "stabilization": {"iterations": 100}}
    }
    """)

    for node_id, data in subgraph.nodes(data=True):
        node_type = data.get("type", "Unknown")
        label = data.get("name", data.get("title", str(node_id)))
        if len(label) > 30:
            label = label[:27] + "..."
        color = NODE_COLORS.get(node_type, "#aaa")
        size = 15 if node_type == "Product" else 25
        net.add_node(
            str(node_id), label=label, title=f"{node_type}: {label}",
            color=color, size=size,
        )

    for src, dst, data in subgraph.edges(data=True):
        net.add_edge(str(src), str(dst), title=data.get("relation", ""), label=data.get("relation", ""))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    net.save_graph(output_path)
    logger.info(f"Interactive graph HTML saved to {output_path} ({subgraph.number_of_nodes()} nodes)")
