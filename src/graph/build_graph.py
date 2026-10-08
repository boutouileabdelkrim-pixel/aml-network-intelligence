"""Build the account-transaction graph from tabular features.

MEMORY-OPTIMIZED: aggregates 5M transactions into unique directed edges
(from_account -> to_account) with weight = number of transactions and
total amount. Produces a NetworkX DiGraph + a parquet edge list.
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import pandas as pd
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TABULAR_PATH = PROJECT_ROOT / "data" / "features" / "tabular.parquet"
GRAPH_DIR = PROJECT_ROOT / "data" / "features"
EDGES_PATH = GRAPH_DIR / "graph_edges.parquet"
GRAPH_PATH = GRAPH_DIR / "graph.gpickle"


def aggregate_edges(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into unique directed edges."""
    logger.info("Aggregating transactions into unique edges ...")
    t0 = time.time()

    edges = (
        df.groupby(["from_account", "to_account"], sort=False)
        .agg(
            n_tx=("amount_paid", "count"),
            total_amount=("amount_paid", "sum"),
            mean_amount=("amount_paid", "mean"),
            max_amount=("amount_paid", "max"),
            n_laundering=("is_laundering", "sum"),
        )
        .reset_index()
    )
    edges["n_tx"] = edges["n_tx"].astype("int32")
    edges["total_amount"] = edges["total_amount"].astype("float32")
    edges["mean_amount"] = edges["mean_amount"].astype("float32")
    edges["max_amount"] = edges["max_amount"].astype("float32")
    edges["n_laundering"] = edges["n_laundering"].astype("int32")
    edges["has_laundering"] = (edges["n_laundering"] > 0).astype("int8")

    logger.info(
        f"Aggregated {len(df):,} tx -> {len(edges):,} unique edges in {time.time()-t0:.1f}s"
    )
    return edges


def build_networkx_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build a directed NetworkX graph from edge list."""
    logger.info("Building NetworkX DiGraph ...")
    t0 = time.time()

    g = nx.DiGraph()
    # Add edges with attributes in bulk
    edge_tuples = list(
        zip(
            edges["from_account"].values,
            edges["to_account"].values,
            edges[["n_tx", "total_amount", "mean_amount", "max_amount", "has_laundering"]].to_dict("records"),
        )
    )
    # NetworkX doesn't accept dict attrs in bulk add_edges_from(tuple3) reliably
    # so we do it manually
    g.add_edges_from(
        [(u, v) for u, v, _ in edge_tuples],
        weight=1,
    )
    # Then set attributes edge by edge (unavoidable)
    for u, v, attrs in edge_tuples:
        g[u][v].update(attrs)

    logger.info(
        f"Graph: {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges in {time.time()-t0:.1f}s"
    )
    return g


def build() -> tuple[Path, Path]:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {TABULAR_PATH} ...")
    df = pd.read_parquet(
        TABULAR_PATH,
        columns=["from_account", "to_account", "amount_paid", "is_laundering"],
        engine="pyarrow",
    )
    logger.info(f"Loaded {len(df):,} transactions")

    edges = aggregate_edges(df)
    del df
    import gc

    gc.collect()

    logger.info(f"Writing edges -> {EDGES_PATH}")
    edges.to_parquet(EDGES_PATH, index=False, engine="pyarrow")

    g = build_networkx_graph(edges)

    logger.info(f"Pickling graph -> {GRAPH_PATH}")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(g, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = GRAPH_PATH.stat().st_size / 1e6
    logger.success(
        f"Graph ready — {g.number_of_nodes():,} nodes, {g.number_of_edges():,} edges ({size_mb:.1f} MB)"
    )
    return EDGES_PATH, GRAPH_PATH


def main() -> None:
    build()


if __name__ == "__main__":
    sys.exit(main())
