"""Topological features per account.

- in/out degree (weighted & unweighted)
- PageRank (scipy-backed)
- Clustering coefficient
- Approximate betweenness (sampled)
- In/out strength (sum of amounts)
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_PATH = PROJECT_ROOT / "data" / "features" / "graph.gpickle"
OUT_PATH = PROJECT_ROOT / "data" / "features" / "graph_topology.parquet"

BETWEENNESS_SAMPLE = 500  # nodes sampled for approximate betweenness


def compute_topology() -> pd.DataFrame:
    logger.info(f"Loading graph from {GRAPH_PATH} ...")
    with open(GRAPH_PATH, "rb") as f:
        g: nx.DiGraph = pickle.load(f)
    logger.info(f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges")

    nodes = list(g.nodes())
    idx = pd.Index(nodes, name="account")

    logger.info("Computing degrees ...")
    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())
    in_deg_w = dict(g.in_degree(weight="n_tx"))
    out_deg_w = dict(g.out_degree(weight="n_tx"))

    # In/out strength = sum of total_amount
    in_strength: dict[str, float] = dict.fromkeys(nodes, 0.0)
    out_strength: dict[str, float] = dict.fromkeys(nodes, 0.0)
    for u, v, data in g.edges(data=True):
        out_strength[u] = out_strength.get(u, 0.0) + float(data.get("total_amount", 0.0))
        in_strength[v] = in_strength.get(v, 0.0) + float(data.get("total_amount", 0.0))

    logger.info("Computing PageRank ...")
    t0 = time.time()
    pagerank = nx.pagerank(g, alpha=0.85, max_iter=100, tol=1e-6, weight="n_tx")
    logger.info(f"PageRank done in {time.time()-t0:.1f}s")

    logger.info("Computing clustering coefficient (undirected) ...")
    t1 = time.time()
    ug = g.to_undirected()
    clustering = nx.clustering(ug)
    del ug
    logger.info(f"Clustering done in {time.time()-t1:.1f}s")

    logger.info(f"Computing approximate betweenness on {BETWEENNESS_SAMPLE} sample ...")
    t2 = time.time()
    # Sample high-degree nodes (more likely on shortest paths)
    top_nodes = sorted(out_deg, key=lambda n: out_deg[n], reverse=True)[:BETWEENNESS_SAMPLE]
    betweenness = nx.betweenness_centrality_subset(
        g, sources=top_nodes, targets=top_nodes, normalized=True
    )
    logger.info(f"Betweenness done in {time.time()-t2:.1f}s")

    logger.info("Assembling DataFrame ...")
    df = pd.DataFrame(
        {
            "in_degree": pd.Series(in_deg, dtype="int32"),
            "out_degree": pd.Series(out_deg, dtype="int32"),
            "in_degree_w": pd.Series(in_deg_w, dtype="int32"),
            "out_degree_w": pd.Series(out_deg_w, dtype="int32"),
            "in_strength": pd.Series(in_strength, dtype="float32"),
            "out_strength": pd.Series(out_strength, dtype="float32"),
            "pagerank": pd.Series(pagerank, dtype="float32"),
            "clustering": pd.Series(clustering, dtype="float32"),
            "betweenness": pd.Series(betweenness, dtype="float32"),
        }
    )
    df = df.reindex(idx)
    df["betweenness"] = df["betweenness"].fillna(0.0)

    # Derived
    df["total_degree"] = (df["in_degree"] + df["out_degree"]).astype("int32")
    df["degree_ratio"] = np.where(
        df["total_degree"] > 0,
        df["in_degree"] / df["total_degree"],
        0.0,
    ).astype("float32")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, engine="pyarrow")
    logger.success(f"Wrote {OUT_PATH} — {df.shape}")
    return df


def main() -> None:
    compute_topology()


if __name__ == "__main__":
    sys.exit(main())
