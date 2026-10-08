"""Community detection with igraph (Louvain) + per-community features.

Outputs per-account:
- community_id (int)
- community_size
- is_in_large_community (size >= threshold)
- account_role_in_community (normalized degree within community)
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import igraph as ig
import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = PROJECT_ROOT / "data" / "features" / "graph.gpickle"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "graph_communities.parquet"

LARGE_COMMUNITY_THRESHOLD = 100


def detect_communities() -> pd.DataFrame:
    logger.info(f"Loading graph from {GRAPH_PATH} ...")
    with open(GRAPH_PATH, "rb") as f:
        g_nx: nx.DiGraph = pickle.load(f)

    logger.info(f"Graph: {g_nx.number_of_nodes():,} nodes, {g_nx.number_of_edges():,} edges")

    logger.info("Converting to igraph (undirected, weighted) ...")
    t0 = time.time()
    # Use undirected for Louvain (standard practice for community detection)
    g_ig = ig.Graph.from_networkx(g_nx)
    g_ig.to_undirected(mode="collapse", combine_edges="sum")
    logger.info(f"igraph: {g_ig.vcount():,} nodes, {g_ig.ecount():,} edges in {time.time()-t0:.1f}s")

    logger.info("Running Louvain community detection ...")
    t1 = time.time()
    partition = g_ig.community_multilevel(weights="n_tx", return_levels=False)
    modularity = partition.modularity
    logger.info(
        f"Louvain done in {time.time()-t1:.1f}s — {len(partition)} communities, modularity={modularity:.4f}"
    )

    # Map: node name -> community id
    node_names = g_ig.vs["_nx_name"] if "_nx_name" in g_ig.vs.attributes() else g_ig.vs["name"]
    membership = pd.Series(partition.membership, index=node_names, name="community_id")
    membership.index.name = "account"

    # Community-level stats
    comm_sizes = membership.value_counts().rename("community_size")
    logger.info(
        f"Community size — min={comm_sizes.min()}, max={comm_sizes.max()}, "
        f"median={int(comm_sizes.median())}"
    )

    df = membership.to_frame().join(comm_sizes, on="community_id")
    df["community_size"] = df["community_size"].astype("int32")
    df["community_id"] = df["community_id"].astype("int32")
    df["is_in_large_community"] = (df["community_size"] >= LARGE_COMMUNITY_THRESHOLD).astype("int8")

    # Compute per-community modularity-like stat: average out-degree inside
    logger.info("Computing per-community aggregate stats ...")
    # For each community, compute the average node degree (undirected)
    node_degrees = pd.Series(g_ig.degree(), index=node_names, dtype="float32")
    node_degrees.name = "community_avg_degree"
    comm_avg_deg = node_degrees.groupby(membership).mean()
    df = df.join(comm_avg_deg.rename("community_avg_degree"), on="community_id")

    # Global modularity as a scalar (same for all rows — for reference)
    df["graph_modularity"] = np.float32(modularity)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, engine="pyarrow")
    logger.success(f"Wrote {OUT_PATH} — {df.shape}")
    return df


def main() -> None:
    detect_communities()


if __name__ == "__main__":
    sys.exit(main())
